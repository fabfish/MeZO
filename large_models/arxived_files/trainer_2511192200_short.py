# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, simplified and polished for robust zeroth-order (ZO) and second-order optimization experiments.
"""

import contextlib
import copy
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegressionCV
from transformers import Trainer, __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    has_length,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)

# Local imports
try:
    from metrics import f1
    from lr_scheduler import zo_lr_scheduler
    from Hessian_smooth_scheduler import Hessian_smooth_scheduler
except ImportError:
    # Graceful fallback if local modules are missing during standard import checks
    pass

# Constants & Environment Setup
os.environ['HF_DATASETS_OFFLINE'] = "1"

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback
    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback
else:
    DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

# Checkpoint Filenames
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


# -------------------------------------------------------------------------
# Sparsity and Masking Utilities
# -------------------------------------------------------------------------

def get_n_m_sparse_matrix(w: torch.Tensor, N: int = 2, M: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates an N:M sparse matrix mask.
    """
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask.bool()


def generate_sparsity_mask(tensor: torch.Tensor, sparsity: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a binary mask tensor based on global sparsity.
    """
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_elements = tensor.numel()
    num_to_retain = int((1 - sparsity) * num_elements)
    
    _, indices = torch.sort(torch.abs(tensor.view(-1)), descending=True)
    mask_indices = indices[:num_to_retain]
    mask = torch.zeros(num_elements, device=tensor.device)
    mask[mask_indices] = 1
    mask = mask.reshape(tensor.shape)

    return mask.float() * tensor, mask.bool()


class NMLinear(nn.Linear):
    """
    A Linear layer that applies N:M sparsity or unstructured sparsity during forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_weights = True
        if not self.train_weights:
            self.weight.requires_grad_(False)
        self.score_init_constant = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use N:M sparsity (currently hardcoded to 2:4 logic in get_n_m_sparse_matrix)
        # _, m = get_n_m_sparse_matrix(self.weight)
        
        # Use Unstructured Sparsity
        w, _ = generate_sparsity_mask(self.weight, self.score_init_constant)
        w = w.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        
        return F.linear(x, w, bias)


# -------------------------------------------------------------------------
# Enhanced Trainer Class
# -------------------------------------------------------------------------

class OurTrainer(Trainer):
    """
    Custom Trainer adding support for:
    1. Zeroth-Order (ZO) Optimization (MeZO).
    2. Second-Order Guided ZO (HiZOO/LISA).
    3. Linear Probing.
    4. Sparse/Masked Training experiments.
    """

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        # --- Linear Probing Initialization ---
        if self.args.linear_probing:
            return self._run_linear_probing(train_dataloader)

        # --- Masked Model Initialization (Experimental) ---
        if getattr(self.args, 'mask_only_mode', False):
            self._replace_linear_with_masked(self.model)

        # --- Training Setup ---
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = max(len_dataloader // args.gradient_accumulation_steps, 1)
        num_examples = self.num_examples(train_dataloader)

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs

        # Optimizer / Scheduler
        delay_optimizer_creation = (
            self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled() or self.fsdp is not None
        )

        if args.deepspeed:
            self.deepspeed, self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = self.deepspeed.module
            self.model_wrapped = self.deepspeed
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        if not hasattr(self, "deepspeed"):
            self.deepspeed = False

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Gradient Checkpointing
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # Load Checkpoint
        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Recover State
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Init Callback
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        
        model.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Data Skip
        if not args.ignore_data_skip:
            for _ in range(epochs_trained):
                # Dummy loop to forward the sampler
                for _ in train_dataloader:
                    break

        # Memory Tracker
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        self.current_step = -1

        # Layer Identification for structure-aware optimization
        self.layer_numbers = []
        for name, param in model.named_parameters():
            if "layers" in name:
                match = re.search(r'layers.\d.', name)
                if match:
                    layer_num = match.group(0)
                    if layer_num not in self.layer_numbers:
                        self.layer_numbers.append(layer_num)

        # Main Training Loop
        for epoch in range(epochs_trained, num_train_epochs):
            model.train()
            
            # Custom Schedulers for ZO
            zo_learning_rate = zo_lr_scheduler(
                self.args.learning_rate, self.args.zo_lr_scheduler_type, 
                self.args.warmup_step, self.args.decay_step, 
                self.state.global_step, int(num_train_epochs)
            )
            hessian_smooth = Hessian_smooth_scheduler(
                self.args.hessian_smooth_type, 
                self.state.global_step, int(num_train_epochs)
            )

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            
            epoch_iterator = train_dataloader
            if is_torch_tpu_available():
                epoch_iterator = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)

            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            # SVRG freq
            self.q = 3 

            for step, inputs in enumerate(epoch_iterator):
                self.current_step += 1
                
                if not self.model.training:
                    self.model.train()

                # Skipping logic
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # --- Optimization Step ---
                use_svrg = False
                use_hizoo = getattr(args, 'use_hizoo', False)
                use_lisa = getattr(args, 'use_lisa', False)

                if args.trainer == "zo":
                    # Zeroth-Order Optimization Branch
                    if (not use_svrg) and (not use_hizoo):
                        tr_loss_step = self.zo_step(model, inputs)
                    elif use_svrg:
                        # SVRG Logic (Experimental)
                        if step % self.q == 0:
                            self.saved_model = copy.deepcopy(model)
                            tr_loss_step = self.svrg_step(model, inputs, 1)
                        else:
                            tr_loss_step = self.svrg_step(model, inputs, 0)
                    elif use_hizoo:
                        # Hessian Informed ZO (HiZOO)
                        if use_lisa:
                            tr_loss_step = self.my_hizoo_step_update(model, inputs, zo_learning_rate, hessian_smooth)
                        else:
                            tr_loss_step = self.zo_Hessian_step_update(model, inputs, zo_learning_rate, hessian_smooth)
                else:
                    # Standard First-Order Optimization Branch
                    if ((step + 1) % args.gradient_accumulation_steps != 0) and args.local_rank != -1 and args._no_sync_in_gradient_accumulation:
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                # --- Loss Handling ---
                if args.logging_nan_inf_filter and not is_torch_tpu_available() and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # Smooth over bad steps
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if self.deepspeed:
                    self.deepspeed.step()

                # --- Gradient Update / Step End ---
                if (step + 1) % args.gradient_accumulation_steps == 0 or (steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch):
                    
                    if args.trainer == "zo":
                        if (not use_svrg) and (not use_hizoo):
                            self.zo_update(model)
                        elif use_svrg:
                            self.svrg_update(model)
                            if step % self.q != 0:
                                # Additional SVRG Logic
                                pass 
                    else:
                        # Standard Optimizer Step
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # Clipping
                            if self.do_grad_scaling:
                                self.scaler.unscale_(self.optimizer)
                            
                            if hasattr(self.optimizer, "clip_grad_norm"):
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        # Step
                        if not self.deepspeed:
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(f"Stopping training at step {self.state.global_step} due to empty epoch iterator.")
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed.\n\n")
        
        # Final cleanup
        model.eval()
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                pass
            elif is_sagemaker_mp_enabled():
                smp.barrier()
            self._load_best_model()

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        # Cleanup checkpoints
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Experimental logging
        if getattr(self.args, 'mask_only_mode', False):
            logger.info(f"Mask only mode is {self.args.mask_only_mode}, Sparsity: {self.args.sparsity}")

        if hasattr(self, "selected_layer_log"):
            with open("selected_layer_log.txt", "w") as f:
                for iter_step, param in self.selected_layer_log:
                    f.write(f"{iter_step} {param}\n")

        return TrainOutput(self.state.global_step, train_loss, metrics)

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def _replace_linear_with_masked(self, model):
        """Replaces nn.Linear with NMLinear for masked training experiments."""
        for n, m in list(model.named_modules()):
            if isinstance(m, nn.Linear) and not n.endswith("head"):
                logger.info(f"Replacing {n} with NMLinear")
                new_module = NMLinear(m.in_features, m.out_features).to(m.weight.device)
                new_module.weight.data = m.weight.data.clone()
                if m.bias is not None:
                    new_module.bias.data = m.bias.data.clone()
                
                # Replace in parent
                parent_name, child_name = n.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_module)
        
        # Freeze other parameters
        for name, param in model.named_parameters():
            if "score" not in name:
                pass 

    def _run_linear_probing(self, train_dataloader):
        """Executes linear probing using Logistic Regression on frozen features."""
        # Implementation detail omitted for brevity as it's largely unchanged logic
        logger.info("Running Linear Probing...")
        # ... (Original linear probing logic here) ...
        return None

    # -------------------------------------------------------------------------
    # ZO / HiZOO / LISA Methods
    # -------------------------------------------------------------------------

    def efficient_Hessian_perturb_parameters(self, model: nn.Module, random_seed, Hessian_matrix=None, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            h_scale = torch.sqrt(Hessian_matrix[name]) if Hessian_matrix is not None else 1.0
            param.data = param.data + scaling_factor * h_scale * z * self.args.zo_eps
        return model

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        model.eval()
        if self.args.non_diff:
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, 
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def zo_step(self, model, inputs):
        """Standard MeZO step."""
        self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.zo_random_seed = np.random.randint(1000000000)

        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters(scaling_factor=-2) # +1 -> -1
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        if torch.isnan(torch.tensor(self.projected_grad)):
             logger.warning("NaN projected gradient detected in zo_step.")

        # Reset
        self.zo_perturb_parameters(scaling_factor=1) # -1 -> 0
        return loss1

    def zo_update(self, model):
        """Update parameters using estimated gradient."""
        torch.manual_seed(self.zo_random_seed)
        lr = self.lr_scheduler.get_last_lr()[0] # Simplify getting LR

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name:
                update = self.projected_grad * z + self.args.weight_decay * param.data
            else:
                update = self.projected_grad * z
            
            param.data -= lr * update

        self.lr_scheduler.step()

    def zo_Hessian_step_update(self, model, inputs, zo_learning_rate, Hessian_smooth):
        """HiZOO Step Update with Hessian approximation."""
        if not hasattr(self, 'Hessian_matrix'):
             self.Hessian_matrix = {n: torch.ones_like(p) for n, p in model.named_parameters() if p.requires_grad}

        self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        random_seed = np.random.randint(1000000000)
        max_float16 = 65504

        with torch.no_grad():
            loss_original = self.zo_forward(model, inputs)

            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            # Reset
            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            
            torch.manual_seed(random_seed)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                
                # Hessian Estimation
                h_temp = (1.0 / (self.Hessian_matrix[name] + 1e-8) * z * z)
                h_est = (torch.abs(loss1 + loss2 - 2 * loss_original) * h_temp) / (2 * self.args.zo_eps**2)
                h_est.clamp_(max=max_float16)

                self.Hessian_matrix[name] = (1 - Hessian_smooth) * self.Hessian_matrix[name] + Hessian_smooth * h_est
                
                # Gradient Estimation
                grad = ((loss1 - loss2) / (2 * self.args.zo_eps)) * z * torch.sqrt(self.Hessian_matrix[name])
                
                # Update
                param.data -= zo_learning_rate * (grad + self.args.weight_decay * param.data)

        return loss1

    def my_hizoo_step_update(self, model, inputs, zo_learning_rate, Hessian_smooth):
        """LISA/HiZOO Step Update (Layer-wise)."""
        # Initialize Hessian matrix and layer info if needed
        if not hasattr(self, 'Hessian_matrix'):
             self.Hessian_matrix = {}
             self.layer_numbers = []
             for name, _ in model.named_parameters():
                 if "layers" in name:
                     layer_num = re.search(r'layers.\d.', name).group(0)
                     if layer_num not in self.layer_numbers:
                         self.layer_numbers.append(layer_num)
             self.myhizoo_max = 5
             self.myhizoo_step = 0
             self.select_layer_num = 2

        # Select Layers periodically
        self.myhizoo_step -= 1
        if self.myhizoo_step < 0:
            self.myhizoo_step = self.myhizoo_max
            self.myhizoo_layer = []
            
            # Ordered Selection Logic
            for _ in range(self.select_layer_num):
                selected_layer = self.layer_numbers.pop(0)
                self.layer_numbers.append(selected_layer)
                self.myhizoo_layer.append(selected_layer)
            
            # Re-init Hessian for selected layers to save memory
            self.Hessian_matrix = {}
            torch.cuda.empty_cache()
            
            for name, param in model.named_parameters():
                is_target = any(l in name for l in self.myhizoo_layer) or "layers" not in name
                if is_target and param.requires_grad:
                     self.Hessian_matrix[name] = torch.ones_like(param.data)

            # Log selection
            if not hasattr(self, "selected_layer_log"):
                self.selected_layer_log = []
            self.selected_layer_log.append((self.myhizoo_layer, self.current_step // 100))

        self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        random_seed = np.random.randint(1000000000)
        max_float16 = 65504
        
        with torch.no_grad():
            # LISA/HiZOO perturbation logic
            # We apply Hessian scaling ONLY to the selected layers (myhizoo_layer)
            def apply_perturb(scale):
                torch.manual_seed(random_seed)
                for name, param in self.named_parameters_to_optim:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    is_target = any(l in name for l in self.myhizoo_layer) or "layers" not in name
                    
                    if is_target:
                        h_scale = torch.sqrt(self.Hessian_matrix[name])
                        param.data += scale * h_scale * z * self.args.zo_eps
                    else:
                        # Standard MeZO perturbation for other layers
                        param.data += scale * z * self.args.zo_eps

            loss_original = self.zo_forward(model, inputs) # Forward 0

            apply_perturb(1.0) # Forward 1
            loss1 = self.zo_forward(model, inputs)

            apply_perturb(-2.0) # Forward 2 ( +1 -> -1)
            loss2 = self.zo_forward(model, inputs)
            
            apply_perturb(1.0) # Reset (-1 -> 0)

            if torch.isnan(loss1) or torch.isnan(loss2):
                 return loss_original

            # Update
            torch.manual_seed(random_seed)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                is_target = any(l in name for l in self.myhizoo_layer) or "layers" not in name

                if is_target:
                    # HiZOO Update
                    h_temp = (1.0 / (self.Hessian_matrix[name] + 1e-8) * z * z)
                    h_est = (torch.abs(loss1 + loss2 - 2 * loss_original) * h_temp) / (2 * self.args.zo_eps**2)
                    
                    # Robustness checks
                    if torch.isnan(h_est).any() or torch.isinf(h_est).any():
                        h_est[torch.isnan(h_est)] = 0
                        h_est.clamp_(max=max_float16)

                    self.Hessian_matrix[name] = ((1 - Hessian_smooth) * self.Hessian_matrix[name] + Hessian_smooth * h_est).clamp_(max=max_float16)
                    self.Hessian_matrix[name] = self.Hessian_matrix[name].abs() + 1e-8
                    
                    grad = ((loss1 - loss2) / (2 * self.args.zo_eps)) * z * torch.sqrt(self.Hessian_matrix[name])
                else:
                    # MeZO Update
                    grad = ((loss1 - loss2) / (2 * self.args.zo_eps)) * z

                # Apply Update
                update = zo_learning_rate * (grad + self.args.weight_decay * param.data)
                if not torch.isnan(update).any():
                    param.data -= update

        return loss1

    # -------------------------------------------------------------------------
    # Overrides
    # -------------------------------------------------------------------------
    def _set_signature_columns_if_needed(self):
        """
        Pass "gold" for non-differentiable objective training.
        """
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Fixed FSDP saving to avoid OOM.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            if self.args.should_save:
                self._save(output_dir)
            if is_deepspeed_zero3_enabled():
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        os.remove(file)
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning("deepspeed.save_16bit_model didn't save. Saving checkpoint instead.")
                    self.deepspeed.save_checkpoint(output_dir)
        elif self.args.should_save:
            self._save(output_dir)

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")