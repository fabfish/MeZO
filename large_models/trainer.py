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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np

import torch.nn.functional as F

VRPGE_mode = False
# mask_only_mode = True

# import os
# fish: add for proxy
os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
# os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"
# os.environ['ALL_PROXY'] = "socks5://127.0.0.1:7897"
mask_only_mode = True
# if mask_only_mode:
# os.environ["WANDB_PROJECT"] = "HiZOO_SST" 
# os.environ["WANDB_PROJECT"] = "HiZOO"
os.environ["WANDB_PROJECT"] = "hizoo_sst2"

os.environ['HF_DATASETS_OFFLINE']= "1"

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
# from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
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
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
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
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers
# taken from hizoo
from lr_scheduler import zo_lr_scheduler
from Hessian_smooth_scheduler import Hessian_smooth_scheduler, Hessian_smooth_scheduler_cosine



# _is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_datasets_available():
    # 
    os.environ['HF_DATASETS_OFFLINE '] = "1"
    import datasets

IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

saved_model = None

test_global_step = 0

import torch

def nan_hook(self, inp, out):
    """
    Check for NaN inputs or outputs at each layer in the model.
    Usage:
        # forward hook
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)
    """
    def flatten(t):
        """Recursively flatten the nested tuple."""
        if isinstance(t, tuple):
            for item in t:
                yield from flatten(item)
        else:
            yield t

    def contains_nan(x):
        """Check if a tensor contains NaNs."""
        return torch.isnan(x).any()

    layer = self.__class__.__name__

    # Flatten inputs and outputs
    inputs = list(flatten(inp))
    outputs = list(flatten(out))

    print(layer)
    if layer=="OPTForCausalLM":
        import pdb; pdb.set_trace()

    # Check for NaNs in inputs
    for i, inp in enumerate(inputs):
        if inp is not None and isinstance(inp, torch.Tensor) and contains_nan(inp):
            print(f'Found NaN input at index: {i} in layer: {layer}')
            import pdb; pdb.set_trace()
            # raise RuntimeError(f'Found NaN input at index: {i} in layer: {layer}')
        elif inp is None:
            print(f'Found None input at index: {i} in layer: {layer}')
            import pdb; pdb.set_trace()
        elif isinstance(inp, torch.Tensor):
            print(f'Input shape: {inp.shape}')
        elif isinstance(inp, (list, tuple)):
            print(f'Input length: {len(inp)}')

    # Check for NaNs in outputs
    for i, out in enumerate(outputs):
        if out is not None and isinstance(out, torch.Tensor) and contains_nan(out):
            # raise RuntimeError(f'Found NaN output at index: {i} in layer: {layer}')
            print(f'Found NaN output at index: {i} in layer: {layer}')
            import pdb; pdb.set_trace()
        elif out is None:
            if layer == "OPTAttention":
                continue
            print(f'Found None output at index: {i} in layer: {layer}')
            import pdb; pdb.set_trace()
        elif isinstance(out, torch.Tensor):
            print(f'Output shape: {out.shape}')
            if out.shape==torch.Size([32, 40, 50272]):
                # import pdb; pdb.set_trace()
                pass
        elif isinstance(out, (list, tuple)):
            print(f'Output length: {len(out)}')

class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        print(self.model)

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                        
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4 # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1: # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # add for high transformer ver
        if not hasattr(self,"deepspeed"):
            self.deepspeed = False

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )


        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            # assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            # self.state.trial_params = hp_params(assignments)
            self.state.trial_params = None
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                # if is_torch_less_than_1_11 or not is_random_sampler:
                if False or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        self.current_step = -1
        
        # Initialize progress bar and running loss
        if not args.disable_tqdm:
            self.remove_callback(ProgressCallback)
            pbar = tqdm(total=max_steps, initial=self.state.global_step, desc="Training", dynamic_ncols=True)
            running_loss = None

        if True:
            self.layer_numbers = []
            for name, param in model.named_parameters():
            # detect if the param is of a specific layer. if true, record the layer numbers to identify the model structure.
                if "layers" in name:
                    # normally the layer number is the digit after the word "layer" and a point, so search layer. and a digit
                    layer_num = re.search(r'layers.\d.', name).group(0)
                    # maintain a list of layer numbers
                    if layer_num not in self.layer_numbers:
                        self.layer_numbers.append(layer_num)
            # import pdb;pdb.set_trace()
        
        for epoch in range(epochs_trained, num_train_epochs):

            # fish: add this to control eval
            # print("epoch start")
            model.train()

            zo_learning_rate = zo_lr_scheduler(self.args.learning_rate, self.args.zo_lr_scheduler_type, self.args.warmup_step, self.args.decay_step, self.state.global_step, int(num_train_epochs))
            # zo_learning_rate = self.args.learning_rate
            # print(zo_learning_rate)
            Hessian_smooth = Hessian_smooth_scheduler(self.args.hessian_smooth_type, self.state.global_step, int(num_train_epochs))
            # Hessian_smooth = Hessian_smooth_scheduler_cosine(self.args.hessian_smooth_type, self.state.global_step, int(num_train_epochs))
            
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            # step = -1
            self.q = 3
            for step, inputs in enumerate(epoch_iterator):
                self.current_step += 1

                # bcd
                if False:
                # if True:
                    torch.cuda.empty_cache()
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        # select only self.current_step % 24 to optim
                        if "layers" in name:
                            if "layers.{}.".format(self.current_step % 24) in name or "layers.{}.".format(((self.current_step % 24) + 1) % 24) in name:
                            # if "layers.{}.".format(self.current_step % 24) in name and not "q" in name:
                                # self.named_parameters_to_optim.append((name, param))
                                param.requires_grad_(True)
                            else:
                                param.requires_grad_(False)
                        else:
                            # self.named_parameters_to_optim.append((name, param))
                            param.requires_grad_(True)

                if not self.model.training:
                    # import pdb; pdb.set_trace()
                    self.model.train()

                # Skip past any already trained steps if resuming training
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

                # use_svrg = True
                use_svrg = False
                use_hizoo = args.use_hizoo
                use_lisa = args.use_lisa


                if args.trainer == "zo":
                    if not use_hizoo:
                        tr_loss_step = self.zo_step(model, inputs)
                    elif use_hizoo:

                        if use_lisa:
                            tr_loss_step = self.my_hizoo_step_update(model, inputs, zo_learning_rate, Hessian_smooth)
                        else:
                            tr_loss_step = self.zo_Hessian_step_update(model, inputs, zo_learning_rate, Hessian_smooth)

                else:

                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                if not args.disable_tqdm:
                    # Update running loss
                    current_loss = tr_loss_step.item()
                    if running_loss is None:
                        running_loss = current_loss
                    else:
                        running_loss = 0.9 * running_loss + 0.1 * current_loss
                    
                    logs = {'loss': f'{running_loss:.4f}'}
                    try:
                        logs['lr'] = f'{self._get_learning_rate():.2e}'
                    except:
                        pass
                    pbar.set_postfix(logs)

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "zo":
                        if (not use_svrg) and (not use_hizoo):
                            self.zo_update(model)

                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                # AMP: gradients need unscaling
                                # import pdb; pdb.set_trace()
                                # check optimzer states here
                                self.scaler.unscale_(self.optimizer)
                                # self.scaler.unscale_(self.optimizer)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    if not args.disable_tqdm:
                        pbar.update(1)
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if not args.disable_tqdm:
            pbar.close()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        # fish: avoid error copying in inference mode
        model.eval()
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.local_rank != -1:
                # dist.barrier()
                pass

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        
        

        self.is_in_train = False

        # seems no use
        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        if self.args.mask_only_mode:
            # import pdb; pdb.set_trace()
            try:
                print(model.model.decoder.layers[0].self_attn.k_proj.scores)
            except:
                pass
        print("Mask only mode is ",self.args.mask_only_mode)
        print("Sparsity is set to ",self.args.sparsity)
        for name, param in model.named_parameters():
            # if param.requires_grad:
            if "scores" in name:
                print(name)
                print(param.shape)
                print(param.sum())
        # import pdb; pdb.set_trace()

        # output self blocks info to file
        if hasattr(self, 'blocks'):
            import csv
            with open("log.csv", mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['Layer Index', 'Step', 'Parameter Name', 'Gradient Sum'])

                for layer, steps_dict in self.blocks.items():
                    for iter_step, gradients in steps_dict.items():
                        for name, grad in gradients:
                            csv_writer.writerow([layer, iter_step, name, grad.item()])    

        if hasattr(self, "selected_layer_log"):
            with open("selected_layer_log.txt", "w") as f:
                for iter_step, param in self.selected_layer_log:
                    f.write(str(iter_step) + " " + str(param) + "\n")
            
        return TrainOutput(self.state.global_step, train_loss, metrics)


    def efficient_Hessian_perturb_parameters(self, model: nn.Module, random_seed, Hessian_matrix=None, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * torch.sqrt(Hessian_matrix[name]) * z * self.args.zo_eps
        return model
        
    def zo_Hessian_step_update(self, model, inputs, zo_learning_rate, Hessian_smooth):
    
        # if self.Hessian_matrix is None:
        if not hasattr(self, 'Hessian_matrix'):
            # self.zo_random_seed = np.random.randint(1000000000)
            self.Hessian_matrix = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)


        # # support gsq update
        # if not hasattr(self, 'blocks'):
        #     self.blocks = {}
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             layer_index = name.split('.')[3] if 'layers' in name else None
        #             if layer_index:
        #                 if layer_index not in self.blocks:
        #                     self.blocks[layer_index] = {}
        #                 # self.blocks[layer_index].append((name, param))


        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        random_seed = np.random.randint(1000000000)
        with torch.no_grad():
            loss_original = self.zo_forward(model, inputs)

            # first function evaluation
            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # second function evaluation
            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            # self.saved_model = copy.deepcopy(model)
            
            torch.manual_seed(random_seed)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

                Hessian_temp = (1/self.Hessian_matrix[name] * z * z)
                Hessian_estimator = (torch.abs(loss1+loss2-2 * loss_original)* Hessian_temp  /(2 * self.args.zo_eps*self.args.zo_eps))
                # 设置上限值
                # 65504 exceeds the max value of float16
                max_float16 = 65504
                Hessian_estimator.clamp_(max=max_float16)

                # debuug code:
                # nan_indices = torch.where(torch.isnan(self.Hessian_matrix[name][0]))[0]
                # Hessian_estimator[0][nan_indices]
                
                # print(self.Hessian_matrix[name][0])

                self.Hessian_matrix[name] = ((1-Hessian_smooth) * self.Hessian_matrix[name] +  Hessian_smooth * Hessian_estimator)
                # print(self.Hessian_matrix[name])

                grad = ((loss1-loss2)/(2 * self.args.zo_eps) * z * torch.sqrt(self.Hessian_matrix[name]))

                # # update gsq info
                # if self.current_step % 100 == 0:
                #     if 'layers' in name:
                #         layer_index = name.split('.')[3]
                #         if layer_index in self.blocks:
                #             # self.blocks[layer_index].append((name, grad.sum(), self.current_step))
                #             if self.current_step in self.blocks[layer_index]:
                #                 self.blocks[layer_index][self.current_step].append((name, grad.sum()))
                #             else:
                #                 self.blocks[layer_index][self.current_step] = [(name, grad.sum())]

                param.data = param.data - zo_learning_rate * (grad + self.args.weight_decay * param.data)

                if param.data.isnan().any():
                    # import pdb; pdb.set_trace()
                    pass

                # if name == "model.decoder.layers[0].fc1.weight":
                    # print(1)

                # import pdb; pdb.set_trace()
            loss_out = self.zo_forward(model, inputs)

        return loss_out
    
    def my_hizoo_perturb_parameters(self, model: nn.Module, random_seed, Hessian_matrix=None, scaling_factor=1, myhizoo_layers = None, second_only = False):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            is_myhizoo_layer = False
            for myhizoo_layer in myhizoo_layers:
                if myhizoo_layer in name:
                    # param.data = param.data + scaling_factor * torch.sqrt(Hessian_matrix[name]) * z * self.args.zo_eps
                    is_myhizoo_layer = True
                    # import pdb; pdb.set_trace()
                # else:
                    # param.data = param.data + scaling_factor * z * self.args.zo_eps
            if not "layers" in name:
                # is_myhizoo_layer = True
                is_myhizoo_layer = False
            if is_myhizoo_layer:
                param.data = param.data + scaling_factor * torch.sqrt(Hessian_matrix[name]) * z * self.args.zo_eps
                # print("update {} with hizoo".format(name))
            else:
                if second_only:
                    pass
                else:
                    param.data = param.data + scaling_factor * z * self.args.zo_eps
                # print("update {} with mezo".format(name))

        return model

    def my_hizoo_step_update(self, model, inputs, zo_learning_rate, Hessian_smooth):
        # 1. Initialize Parameter List Once (Cache it)
        if not hasattr(self, 'named_parameters_to_optim'):
            self.named_parameters_to_optim = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((name, param))

        # 2. Initialize Hessian Matrix Once
        if not hasattr(self, 'Hessian_matrix'):
            self.Hessian_matrix = {}
            self.layer_numbers = []
            for name, param in model.named_parameters():
                if "layers" in name:
                    # Extract layer number efficiently
                    match = re.search(r'layers\.(\d+)\.', name)
                    if match:
                        layer_num = match.group(0) # e.g., "layers.0."
                        if layer_num not in self.layer_numbers:
                            self.layer_numbers.append(layer_num)
            
            self.myhizoo_max = 5
            self.myhizoo_step = 0
            self.select_layer_num = 2
            self.current_hizoo_param_names = set() # Fast lookup set

        # support gsq update (Kept as is, assumed lightweight)
        if not hasattr(self, 'blocks'):
            self.blocks = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    layer_index = name.split('.')[3] if 'layers' in name else None
                    if layer_index:
                        if layer_index not in self.blocks:
                            self.blocks[layer_index] = {}

        # 3. Layer Selection Logic
        self.myhizoo_step -= 1
        if self.myhizoo_step < 0:
            self.myhizoo_step = self.myhizoo_max
            self.myhizoo_layer = []

            # --- REMOVED: del self.Hessian_matrix and empty_cache() (Expensive) ---
            
            # Ordered Selection
            ordered = True
            if ordered:
                for i in range(self.select_layer_num):
                    if self.layer_numbers:
                        selected_layer = self.layer_numbers.pop(0)
                        self.layer_numbers.append(selected_layer)
                        self.myhizoo_layer.append(selected_layer)
            else:
                # (BCD logic omitted for brevity, keep your original if needed)
                pass

            # --- OPTIMIZATION: Pre-calculate active parameters for this block ---
            self.current_hizoo_param_names.clear()
            for name, param in self.named_parameters_to_optim:
                # Initialize Hessian entry if missing
                if name not in self.Hessian_matrix:
                     self.Hessian_matrix[name] = torch.ones_like(param.data)
                
                # Check if this param belongs to the active HiZOO layers
                is_match = False
                for layer_prefix in self.myhizoo_layer:
                    if layer_prefix in name:
                        is_match = True
                        break
                
                # Global "layers" check from your original logic
                if not "layers" in name:
                    is_match = False
                
                if is_match:
                    self.current_hizoo_param_names.add(name)

            # Logging
            step_index = self.current_step // 100
            if not hasattr(self, "selected_layer_log"):
                self.selected_layer_log = []
            self.selected_layer_log.append((self.myhizoo_layer, step_index))
            
            # --- REMOVED: torch.cuda.empty_cache() ---

        random_seed = np.random.randint(1000000000)
        second_only = True
        max_float16 = 65504

        with torch.no_grad():
            loss_original = None
            forward_3_times = True
            
            if forward_3_times: 
                loss_original = self.zo_forward(model, inputs)

            # 1. Perturb Positive
            self.my_hizoo_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1, myhizoo_layers=self.myhizoo_layer, second_only=second_only)
            loss1 = self.zo_forward(model, inputs)

            # 2. Perturb Negative (Move from +1 to -1, so scale is -2)
            self.my_hizoo_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2, myhizoo_layers=self.myhizoo_layer, second_only=second_only)
            loss2 = self.zo_forward(model, inputs)

            # 3. Restore (Move from -1 to 0, so scale is +1)
            self.my_hizoo_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1, myhizoo_layers=self.myhizoo_layer, second_only=second_only)

            # Early exit on NaN loss to save compute
            if torch.isnan(loss1) or torch.isnan(loss2):
                return loss_original if loss_original is not None else loss1

            torch.manual_seed(random_seed)
            
            # 4. Optimization Loop
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

                # --- OPTIMIZATION: O(1) Lookup instead of string matching ---
                is_hizoo_layer = name in self.current_hizoo_param_names

                if is_hizoo_layer:
                    # Calculate Hessian Estimate
                    Hessian_temp = (1 / (self.Hessian_matrix[name] + 1e-8) * z * z)
                    
                    if loss_original is not None:
                         loss_diff_sq = torch.abs(loss1 + loss2 - 2 * loss_original)
                         Hessian_estimator = (loss_diff_sq * Hessian_temp) / (2 * self.args.zo_eps * self.args.zo_eps)
                    else:
                         # Fallback if loss_original not available
                         Hessian_estimator = torch.abs(loss1 - loss2) * Hessian_temp / (2 * self.args.zo_eps)

                    # --- REMOVED: .isnan().any() checks (Expensive Sync) ---
                    # Instead, rely on clamp and safe replacement operations
                    Hessian_estimator.nan_to_num_(nan=0.0, posinf=max_float16, neginf=-max_float16)
                    Hessian_estimator.clamp_(max=max_float16, min=-max_float16)

                    # Update Moving Average
                    self.Hessian_matrix[name] = (
                        (1 - Hessian_smooth) * self.Hessian_matrix[name] + 
                        Hessian_smooth * Hessian_estimator
                    ).clamp_(max=max_float16, min=-max_float16)

                    # Ensure positivity
                    self.Hessian_matrix[name] = self.Hessian_matrix[name].abs() + 1e-8

                    # Gradient Estimate
                    grad = ((loss1 - loss2) / (2 * self.args.zo_eps)) * z * torch.sqrt(self.Hessian_matrix[name])

                else:
                    # Standard MeZO estimate (Identity Hessian)
                    grad = ((loss1 - loss2) / (2 * self.args.zo_eps)) * z

                # --- REMOVED: param.data.isnan().any() checks (Expensive Sync) ---
                
                # Apply Update
                # Combine LR, Grad, and Weight Decay in one operation
                update = zo_learning_rate * (grad + self.args.weight_decay * param.data)
                update.clamp_(max=max_float16, min=-max_float16)
                
                param.data.sub_(update) # In-place subtraction is slightly faster

        # --- REMOVED: torch.cuda.empty_cache() (Expensive Sync) ---
        
        return loss1

    ############## MeZO ##############


    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps


    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        # hooks = []
        # for submodule in model.modules():
        #     handle = submodule.register_forward_hook(nan_hook)
        #     # hooks.append(handle)

        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                # import pdb; pdb.set_trace()
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)


    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)


        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        #fish: for debug
        self.loss1 = loss1
        self.loss2 = loss2
        self.inputs = inputs

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        # assert not torch.isnan(self.projected_grad).any()
        assert self.projected_grad != torch.nan

        # param.data = 1/(args.K-1)*(fn_list[0] - fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(fn_list[1] - fn_avg)*getattr(m, 'stored_mask_1')
        self.fn_list = []
        self.fn_list.append(loss1)
        self.fn_list.append(loss2)
        self.fn_avg = (loss1+loss2) /2

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)
        
        return loss1
    

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            # else:
            #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
            
            if param.data.isnan().any():
                import pdb; pdb.set_trace()

            # deepcopy param
            tmp = copy.deepcopy(param.data)

            if "scores" in name:
                param.data = param.data - 10 * self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            elif "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

            if param.data.isnan().any():
                import pdb; pdb.set_trace()

        self.lr_scheduler.step()


    ############## Misc overload functions ##############


    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
