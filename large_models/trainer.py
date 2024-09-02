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
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"
os.environ['ALL_PROXY'] = "socks5://127.0.0.1:7897"
mask_only_mode = True
# if mask_only_mode:
# os.environ["WANDB_PROJECT"] = "HiZOO_SST" 
# os.environ["WANDB_PROJECT"] = "HiZOO"
os.environ["WANDB_PROJECT"] = "hizoo_sst2"

os.environ['HF_DATASETS_OFFLINE']= "1"

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

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
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
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
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
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
from transformers.utils.generic import ContextManagers
# taken from hizoo
from lr_scheduler import zo_lr_scheduler
from Hessian_smooth_scheduler import Hessian_smooth_scheduler, Hessian_smooth_scheduler_cosine



_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    # 
    os.environ['HF_DATASETS_OFFLINE '] = "1"
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

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


############################################
# following modules are for masked finetuning
# masked bert: normal optimizer, sample mask + finetune
# vrpge: vrpge update, mask + finetune
# mask only vrpge: mask tuning
# mezo: fo ft
# mask only mezo: mask tuning
class _Binarizer3(torch.autograd.Function):
    # method based on masking as an efficient alternative to finetuning
    @staticmethod
    def forward(ctx, inputs):
        return torch.bernoulli(inputs).bool()

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput
    
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, mask=None):
        ctx.with_mask = False
        if mask is not None:
            weight = weight * torch.bernoulli(mask)

            # weight = weight * torch.bernoulli(torch.sigmoid(mask))

            # note that mask is of low precision
            # mask f16, weight f32
            # import pdb; pdb.set_trace()
            ctx.with_mask = True

        ctx.save_for_backward(weight, bias, x)
        ctx.weight_dtype = weight.dtype
        ctx.mask_dtype = mask.dtype if mask is not None else None
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = grad_mask =  None

        tensors = ctx.saved_tensors
        weight, bias, input = tensors
        # 32, 32, 16

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))
        # else:
            # the first layer
            # import pdb; pdb.set_trace()

        if ctx.needs_input_grad[1]:

            grad_weight = grad_output.transpose(-2, -1).matmul(input).to(dtype=grad_output.dtype)

            if ctx.with_mask and ctx.needs_input_grad[3]:

                # grad_mask = grad_output.transpose(-2, -1).matmul(input.to(dtype=ctx.mask_dtype))
                grad_mask = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        if ctx.with_mask and ctx.needs_input_grad[3]:

            if grad_mask is None:
                # grad_mask = grad_output.transpose(-2, -1).matmul(input.to(dtype=ctx.mask_dtype))
                grad_mask = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        # import pdb; pdb.set_trace()

        # return grad_input, grad_weight, grad_bias, grad_mask
        # if scores is half, the output will be
        # f16, None, f16, f16

        # unknown amp scaler error /home/*/anaconda3/lib/python3.10/site-packages/torch/cuda/amp/grad_scaler.py", line 258
        # return grad_input.float(), grad_weight, grad_bias.float(), grad_mask.float()
        return grad_input, grad_weight, grad_bias, grad_mask
    
class StraightThroughBinomialSampleNoGrad(torch.autograd.Function):
    # method took from vrpge
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)

class MaskedLinear(nn.Linear):
    '''
    update: gradient based
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, sparsity = 0.95):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()).to(dtype=torch.float16))
        self.train_weights = False
        if self.train_weights == False:
            self.weight.requires_grad_(False)
        
        # print(sparsity)
        self.score_init_constant = sparsity
        # it is important to set to half precision
        self.scores.data = (
            # torch.ones_like(self.scores).half() * self.score_init_constant
            torch.ones_like(self.scores) * self.score_init_constant
        )
        # print(self.scores)

    @property
    def clamped_scores(self):
        return self.scores

    def forward(self, x):
        # if self.training:
        if True:
        #     # logger.info("eval mode")
        #     w = (self.weight > self.score_init_constant).float()
        #     w = self.weight * w
        #     x = F.linear(x, w, self.bias)
        # else:
            # logger.info("train mode")
            # keep scores > 0 
            self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)
            # self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
            # self.subnet = _Binarizer3.apply(self.scores)

            # w = self.weight * self.subnet
            # w = w.to(self.weight.dtype)
            # x = x.to(self.weight.dtype)
            try:
                # x = F.linear(x, w, self.bias)
                x = linear.apply(x, self.weight, self.bias, self.scores)
            except:
                import pdb; pdb.set_trace()

            # logger.info("train mode")

        else:
            

            # w = self.weight * self.subnet
            # w = (self.weight > self.score_init_constant).float()
            # w = self.weight * w

            # self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)

            self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)
            w = self.weight * torch.bernoulli(self.scores)

            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)

        return x
    
class MaskedLinearS(nn.Linear):
    '''
    update: gradient based
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, sparsity = 0.95):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.train_weights = False
        if self.train_weights == False:
            self.weight.requires_grad_(False)
        
        # print(sparsity)
        self.score_init_constant = sparsity
        # it is important to set to half precision
        self.scores.data = (
            # torch.ones_like(self.scores).half() * self.score_init_constant
            torch.ones_like(self.scores) * self.score_init_constant
        )
        # print(self.scores)

    @property
    def clamped_scores(self):
        return self.scores

    def forward(self, x):
        # if self.training:
        if True:
        #     # logger.info("eval mode")
        #     w = (self.weight > self.score_init_constant).float()
        #     w = self.weight * w
        #     x = F.linear(x, w, self.bias)
        # else:
            # logger.info("train mode")
            # keep scores > 0 
            self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)
            # self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
            # self.subnet = _Binarizer3.apply(self.scores)

            # w = self.weight * self.subnet
            # w = w.to(self.weight.dtype)
            # x = x.to(self.weight.dtype)

            try:
                # x = F.linear(x, w, self.bias)
                x = linear.apply(x, self.weight, self.bias, self.scores)
            except:
                import pdb; pdb.set_trace()

            # logger.info("train mode")
            # x = linear.apply(x, self.weight, self.bias, self.scores)

            # w = self.weight * torch.bernoulli(self.scores)
            # x = F.linear(x, self.weight, self.bias)

        else:
            

            # w = self.weight * self.subnet
            # w = (self.weight > self.score_init_constant).float()
            # w = self.weight * w

            # self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)

            self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)
            w = self.weight * torch.bernoulli(self.scores)

            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)

        return x

def get_n_m_sparse_matrix(w, N=2, M=4):
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask.bool()

def generate_sparsity_mask(tensor, sparsity):
    """
    Generate a binary mask tensor by sorting tensor elements and masking out the smallest ones
    based on the given sparsity, and move it to the specified device.
    
    Parameters:
        tensor (torch.Tensor): The input tensor.
        sparsity (float): The sparsity level (between 0 and 1), where 1 means all zeros.
        device (str): The device identifier where the mask will be placed (default 'cuda:0').

    Returns:
        torch.Tensor: A binary mask tensor with the same shape as the input tensor, on the specified device.
    """
    # 确保稀疏度在有效范围内
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be between 0 and 1.")
    
    # 检查设备是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    
    # 确定保留元素的数量
    num_elements = tensor.numel()
    num_to_retain = int((1 - sparsity) * num_elements)
    
    # 排序元素并生成掩码
    _, indices = torch.sort(torch.abs(tensor.view(-1)), descending=True)
    mask_indices = indices[:num_to_retain]  # 只保留绝对值最大的元素索引
    mask = torch.zeros(num_elements, device=tensor.device)
    mask[mask_indices] = 1
    mask = mask.reshape(tensor.shape)

    return mask.float() * tensor, mask.bool()

class NMlinearfunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask):
        ctx.save_for_backward(input, weight, bias, mask)
        return F.linear(input, mask * weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.matmul(weight.t())
            grad_input = grad_output.matmul((mask*weight).t())
        if ctx.needs_input_grad[1]:
            grad_weight = mask * (grad_output.t().matmul(input))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        if ctx.needs_input_grad[3]:
            import pdb; pdb.set_trace()
            grad_mask = grad_output.t().matmul(input)
        return grad_input, grad_weight, grad_bias, grad_mask
    
from torch.sparse import to_sparse_semi_structured
class NMLinear(nn.Linear):
    # n=2 m=4 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_weights = True
        if self.train_weights == False:
            self.weight.requires_grad_(False)
        self.score_init_constant = 0.5

    def forward(self, x):
        # if self.training:
        if True:
            # _, m = get_n_m_sparse_matrix(self.weight)
            _, m = get_n_m_sparse_matrix(self.weight)
            # w device is cuda
            # w = nn.Parameter(to_sparse_semi_structured(self.weight.masked_fill(~m, 0)))
            # took 50% more time
            w, m = generate_sparsity_mask(self.weight, self.score_init_constant)
            w = w.to(x.dtype)
            # import pdb; pdb.set_trace()
            self.bias = self.bias.to(x.dtype)
            # m = m.to(x.dtype)
            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)
            # no need
            # x = NMlinearfunction.apply(x, w, self.bias, m)
        else:
            w = self.weight * self.subnet
            w = w.to(x.dtype)
            self.bias = self.bias.to(x.dtype)
            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)

        return x

class VRPGEMaskedLinear(nn.Linear):
    def __init__(self, sparsity = 0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = True
        if self.train_weights == False:
            self.weight.requires_grad_(False)
        self.score_init_constant = sparsity
        self.scores.data = (
                torch.ones_like(self.scores) * self.score_init_constant
        )
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))
        self.j = 0
        self.subnet = torch.zeros_like(self.scores)

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        # if True:
        self.scores.data = torch.clamp(self.scores.data, min=0.0, max=1.0)
        self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
        # if self.training:
        if True:
            # keep scores > 0 
            
            # self.subnet = _Binarizer3.apply(self.scores)
            if self.j == 0:
                self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                self.j = 1
            else:
                self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                self.j = 0
            w = self.weight * self.subnet
            w = w.to(x.dtype)
            self.bias = self.bias.to(x.dtype)
            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)
        else:
            w = self.weight * self.subnet
            w = w.to(x.dtype)
            self.bias = self.bias.to(x.dtype)
            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)

        return x
    
class VRPGELinear(nn.Linear):
    def __init__(self, sparsity = 0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        # self.register_buffer('subnet', torch.zeros_like(self.scores))
        # self.train_weights = True
        # if self.train_weights == False:
        #     self.weight.requires_grad_(False)
        # self.score_init_constant = sparsity
        # self.scores.data = (
        #         torch.ones_like(self.scores) * self.score_init_constant
        # )
        self.register_buffer("stored_mask_0", torch.zeros_like(self.weight))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.weight))
        self.j = 0
        # self.subnet = torch.zeros_like(self.weight)

    def forward(self, x):
        # if self.training:
        if True:
            # keep scores > 0 
            if self.j == 0:
                self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                self.j = 1
            else:
                self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                self.j = 0
            w = self.weight.to(x.dtype)
            self.bias = self.bias.to(x.dtype)
            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)
        else:
            w = self.weight * self.subnet
            w = w.to(x.dtype)
            self.bias = self.bias.to(x.dtype)
            # x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = F.linear(x, w, self.bias)

        return x
    


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

        # Do the layer replacing before all the change on the model

        print("Mask only mode is ",self.args.mask_only_mode)
        print("Sparsity is set to ",self.args.sparsity)
        print(self.model)
        ## VRPGE
        mask_mode = self.args.mask_only_mode
        # VRPGE_mode=False
        # import pdb; pdb.set_trace()
        if mask_mode == True:
            model = self.model
            # Create a list of items to iterate over
            items = list(model.named_modules())

            for n, m in items:
                if isinstance(m, nn.Linear):
                    # if n.endswith("query") or n.endswith("value") or n.endswith("attention.output.dense"):
                    # if "attention" in n or "output" in n:
                    # if "attention" in n:
                    # if n.endswith("attention.output.dense"):
                    # if n.endswith("out_proj") or n.endswith("k_proj") or n.endswith("v_proj") or n.endswith("fc1") or n.endswith("fc2"):
                    # if True:
                    if not n.endswith("head"):
                    # if n.endswith("v_proj"):
                        print("replacing {}".format(n))
                        # new_module = LinearSparse(m.in_features, m.out_features).to(self.model.device)
                        # new_module = IA3Layer(m.in_features, m.out_features).to(self.model.device)
                        # new_module = VRPGEMaskedLinear(self.args.sparsity, m.in_features, m.out_features).to(self.model.device)
                        new_module = NMLinear(m.in_features, m.out_features).to(self.model.device)
                        # new_module = MaskedLinear(m.in_features, m.out_features, sparsity=self.args.sparsity).to(self.model.device)
                        # Copy the weights and biases
                        new_module.weight.data = m.weight.data.clone()
                        if m.bias is not None:
                            new_module.bias.data = m.bias.data.clone()
                        # Replace the module
                        name_parts = n.split('.')
                        parent = model
                        for part in name_parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, name_parts[-1], new_module)

            # import pdb; pdb.set_trace()
            for name, param in model.named_parameters():
                # if (not "lm" in name) and (not "embedding" in name) and (not "scores"  in name):
                # if not "lm" in name:
                if not "score" in name:
                # if not "embedding" in name:
                # if True:
                # if "proj" in name or "fc" in name or "layer_norm" in name:
                    # param.requires_grad_(False)
                    pass

            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)

            # import pdb; pdb.set_trace()
            self.model = model            



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
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
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
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # fish: add memory tracker
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # for submodule in model.modules():
        #     if isinstance(submodule, nn.Linear) or isinstance(submodule, nn.Embedding) or isinstance(submodule, nn.ReLU) or isinstance(submodule, nn.LayerNorm) or isinstance(submodule, nn.Dropout):    
        #         submodule.register_forward_hook(nan_hook)
        #         print("Register hook for {}".format(submodule))

        self.current_step = -1
        
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
            # print("epoch start, model.train is", self.model.training)
            if not self.model.training:
                import pdb; pdb.set_trace()

            zo_learning_rate = zo_lr_scheduler(self.args.learning_rate, self.args.zo_lr_scheduler_type, self.args.warmup_step, self.args.decay_step, self.state.global_step, int(num_train_epochs))
            # zo_learning_rate = self.args.learning_rate
            # print(zo_learning_rate)
            Hessian_smooth = Hessian_smooth_scheduler(self.args.hessian_smooth_type, self.state.global_step, int(num_train_epochs))
            # Hessian_smooth = Hessian_smooth_scheduler_cosine(self.args.hessian_smooth_type, self.state.global_step, int(num_train_epochs))
            
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
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

                if False:
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        # select only self.current_step % 24 to optim
                        if "layers" in name:
                            if "layers.{}.".format(self.current_step % 24) in name or "layers.{}.".format(((self.current_step % 24) + 1) % 24) in name:
                                # self.named_parameters_to_optim.append((name, param))
                                param.requires_grad_(True)
                            else:
                                param.requires_grad_(False)
                        else:
                            # self.named_parameters_to_optim.append((name, param))
                            param.requires_grad_(True)

                # test_global_step = step
                # if self.current_step==2563:
                if False:
                    for submodule in model.modules():
                        submodule.register_forward_hook(nan_hook)
                    # import pdb;pdb.set_trace()

                # model.train()
                # print("epoch start, model.train is", self.model.training)
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
                # use_hizoo = True
                # use_hizoo = False
                use_hizoo = args.use_hizoo
                use_lisa = args.use_lisa

                # move into func
                # if use_hizoo:
                #     # self.zo_random_seed = np.random.randint(1000000000)
                #     self.Hessian_matrix = {}
                #     for name, param in model.named_parameters():
                #         if param.requires_grad:
                #             self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                # MeZO added: estimate gradient
                if args.trainer == "zo":
                    pass
                else:
                    pass
                    # for name, param in model.named_parameters():
                    #     if not ("23" in name) and not ("embed" in name) and not ("head" in name):
                    #         param.requires_grad_(False)

                if args.trainer == "zo":
                    if (not use_svrg) and (not use_hizoo):
                        tr_loss_step = self.zo_step(model, inputs)
                    elif use_svrg:
                    # use svrg
                        self.named_parameters_to_optim = []
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                self.named_parameters_to_optim.append((name, param))
                        if step % self.q == 0:
                            # save model weights
                            saved_model = copy.deepcopy(model)
                            self.saved_model = saved_model
                            tr_loss_step = self.svrg_step(model, inputs, 1)
                        else:
                            tr_loss_step = self.svrg_step(model, inputs, 0)
                            # self.zo_step(self.saved_model, inputs)
                    elif use_hizoo:
                        # print(1)
                        # tr_loss_step = self.zo_Hessian_step_update(model, inputs, zo_learning_rate, Hessian_smooth)
                        # tr_loss_step = self.myH(model, inputs, zo_learning_rate, Hessian_smooth)
                        # tr_loss_step = self.my_hizoo_step_update(model, inputs, zo_learning_rate, Hessian_smooth)

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

                        # # model.train()
                        # # print("epoch start, model.train is", self.model.training)
                        # if not self.model.training:
                        #     import pdb; pdb.set_trace()

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
                            # self.zo_vrpge_update(model)
                        elif use_svrg:

                            self.svrg_update(model)

                            if step % self.q != 0:
                                self.named_parameters_to_optim = []
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        self.named_parameters_to_optim.append((name, param))
                                self.saved_parameters = self.named_parameters_to_optim
                                self.svrg_step(self.saved_model, inputs, 0)
                                self.named_parameters_to_optim = self.saved_parameters
                                self.svrg_update(model, 0)

                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                # import pdb; pdb.set_trace()
                                # check optimzer states here
                                self.scaler.unscale_(self.optimizer)
                                # self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
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

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        # fish: avoid error copying in inference mode
        model.eval()
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                # dist.barrier()
                pass
            elif is_sagemaker_mp_enabled():
                smp.barrier()

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
    
        # import pdb; pdb.set_trace()
        # if self.Hessian_matrix is None, this will be the initialization
        if not hasattr(self, 'Hessian_matrix'):
            # self.zo_random_seed = np.random.randint(1000000000)
            self.Hessian_matrix = {}
            self.layer_numbers = []
            for name, param in model.named_parameters():
                # detect if the param is of a specific layer. if true, record the layer numbers to identify the model structure.
                if "layers" in name:
                    # normally the layer number is the digit after the word "layer" and a point, so search layer. and a digit
                    layer_num = re.search(r'layers.\d.', name).group(0)
                    # maintain a list of layer numbers
                    if layer_num not in self.layer_numbers:
                        self.layer_numbers.append(layer_num)
                # if param.requires_grad:
                    # self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # initialize a step count for hizoo layers
            # interval
            self.myhizoo_max = 5
            self.myhizoo_step = 0 # count from 10 to 0
            # self.select_layer_num = len(self.layer_numbers)
            self.select_layer_num = 2

        # support gsq update
        if not hasattr(self, 'blocks'):
            self.blocks = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    layer_index = name.split('.')[3] if 'layers' in name else None
                    if layer_index:
                        if layer_index not in self.blocks:
                            self.blocks[layer_index] = {}
                        # self.blocks[layer_index].append((name, param))

        # after initialization, first we try the badam way of coordinate iterative optimization
        # each myhizoo turn, we do myhizoo_max steps of hybrid mezo and hizoo
        # with hizoo on the chosen layer (iteratively)
        self.myhizoo_step -=1
        if self.myhizoo_step < 0:
            # if we finish the myhizoo_max steps, we reset the step count and choose the next layer
            self.myhizoo_step = self.myhizoo_max
            # choose the next layer
            self.myhizoo_layer = []


            del self.Hessian_matrix
            torch.cuda.empty_cache()
            # self.zo_random_seed = np.random.randint(1000000000)
            self.Hessian_matrix = {}
            # ordered = False
            ordered = True
            if ordered:
                # Ordered Selection
                for i in range(self.select_layer_num):
                    # self.myhizoo_layer = self.layer_numbers.pop(0)
                    selected_layer = self.layer_numbers.pop(0)
                    # change to random selection
                    # selected_layer = self.layer_numbers.pop(np.random.randint(len(self.layer_numbers)))
                    # put the layer number back to the end of the list
                    self.layer_numbers.append(selected_layer)
                    # print the layer number
                    # print(self.myhizoo_layer)
                    # self.myhizoo_layer = "layer." + self.myhizoo_layer
                    self.myhizoo_layer.append(selected_layer)

            else:
                # BCD: Gauss Southwell Quadratic (Diagnal) Selection
                step_index = self.current_step // 100
                if hasattr(self, "blocks"):
                    max_score = -1
                    selected_layer = 0
                    for layer_index in self.blocks:
                        tmp_score = 0
                        if step_index in self.blocks[layer_index]:
                            # sum self.blocks[layer_index][step_index]
                            for name, value in self.blocks[layer_index][step_index]:
                                tmp_score += value
                            if tmp_score >= max_score:
                                selected_layer = layer_index
                        elif step_index-1 in self.blocks[layer_index]:
                            for name, value in self.blocks[layer_index][step_index]:
                                tmp_score += value
                            if tmp_score >= max_score:
                                selected_layer = layer_index
                    # to str
                    selected_layer = "layers." + str(selected_layer)
                    self.myhizoo_layer.append(selected_layer)

            step_index = self.current_step // 100
            if not hasattr(self,"selected_layer_log"):
                self.selected_layer_log = []
            # log selected and step
            self.selected_layer_log.append((selected_layer, step_index))

            # print(self.myhizoo_layer)
            for name, param in model.named_parameters():
                for myhizoo_layer in self.myhizoo_layer:
                    if myhizoo_layer in name:
                        if param.requires_grad:
                            self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if not "layers" in name:
                    if param.requires_grad:
                        self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)


        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        random_seed = np.random.randint(1000000000)

        second_only = True
        max_float16 = 65504

        with torch.no_grad():
            loss_original = None
            # forward_3_times = False
            forward_3_times = True
            if forward_3_times: 
                loss_original = self.zo_forward(model, inputs)

            # first function evaluation
            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            model = self.my_hizoo_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1, myhizoo_layers=self.myhizoo_layer, second_only=second_only)
            loss1 = self.zo_forward(model, inputs)


            # second function evaluation
            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2)
            model = self.my_hizoo_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2, myhizoo_layers=self.myhizoo_layer, second_only=second_only)
            loss2 = self.zo_forward(model, inputs)

            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            model = self.my_hizoo_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1, myhizoo_layers=self.myhizoo_layer, second_only=second_only)
            # self.saved_model = copy.deepcopy(model)
            
            torch.manual_seed(random_seed)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

                # if self.myhizoo_layer in name:
                is_hizoo_layer = False
                for myhizoo_layer in self.myhizoo_layer:
                    if myhizoo_layer in name:
                        is_hizoo_layer = True
                    if not "layers" in name:
                        # is_hizoo_layer = True
                        is_hizoo_layer = False
                if is_hizoo_layer:
                    Hessian_temp = (1/self.Hessian_matrix[name] * z * z)
                    if loss_original is not None:
                        Hessian_estimator = (torch.abs(loss1+loss2-2 * loss_original)* Hessian_temp  /(2 * self.args.zo_eps*self.args.zo_eps))
                    else:
                        Hessian_estimator = torch.abs(loss1 - loss2) * Hessian_temp  /(2 * self.args.zo_eps)
                    # 设置上限值
                    # 65504 exceeds the max value of float16
                    Hessian_estimator.clamp_(max=max_float16, min=-max_float16) # comment for bf16

                    # debuug code:
                    # nan_indices = torch.where(torch.isnan(self.Hessian_matrix[name][0]))[0]
                    # Hessian_estimator[0][nan_indices]
                    
                    # print(self.Hessian_matrix[name][0])

                    # self.Hessian_matrix[name] = ((1-Hessian_smooth) * self.Hessian_matrix[name] +  Hessian_smooth * Hessian_estimator)
                    self.Hessian_matrix[name] = ((1-Hessian_smooth) * self.Hessian_matrix[name] +  Hessian_smooth * Hessian_estimator).clamp_(max=max_float16, min=-max_float16)
                    # print(self.Hessian_matrix[name])

                    grad = ((loss1-loss2)/(2 * self.args.zo_eps) * z * torch.sqrt(self.Hessian_matrix[name]))

                else:
                    # grad = ((loss1-loss2)/(2 * self.args.zo_eps) * z)
                    # grad = torch.Tensor([0.0]).to(param.device)
                    grad  = loss1-loss1


                if param.data.isnan().any():
                    import pdb; pdb.set_trace()
                    # pass
                
                tmp = param.data - zo_learning_rate * (grad + self.args.weight_decay * param.data)
                tmp.clamp_(max=max_float16, min=-max_float16)
                

                if tmp.isnan().any():
                    import pdb; pdb.set_trace()
                    # pass

                else:
                    param.data = tmp

                if name == "model.decoder.layers.0.fc1.weight":
                    # print(1)
                    # import pdb; pdb.set_trace()
                    pass

                # import pdb; pdb.set_trace()
            # loss_out = self.zo_forward(model, inputs)


        torch.cuda.empty_cache()
        return loss1

    def myH(self, model, inputs, zo_learning_rate, Hessian_smooth):
    
        # if self.Hessian_matrix is None:
        if not hasattr(self, 'Hessian_matrix'):
            # self.zo_random_seed = np.random.randint(1000000000)
            self.Hessian_matrix = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        random_seed = np.random.randint(1000000000)
        random_seed2 = np.random.randint(1000000000)
        with torch.no_grad():
            # loss_original = self.zo_forward(model, inputs)
            self.zo_perturb_parameters(random_seed, scaling_factor=1)

            # first function evaluation
            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            
            loss1 = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(random_seed2, scaling_factor=1)

            loss12 = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(random_seed, scaling_factor=-2)
            loss22 = self.zo_forward(model, inputs)

            # second function evaluation
            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2)
            self.zo_perturb_parameters(random_seed2, scaling_factor=-1)
            loss2 = self.zo_forward(model, inputs)

            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            # self.saved_model = copy.deepcopy(model)
            
            def vector_divide(s, t):
                # s is a vector, t is a vector
                # The output should be a matrix where each column i of the output matrix is s divided by t[i]
                s = s.view(1, -1)
                t = t.view(-1, 1)  # Reshape t to a column vector
                S = s / t  # Broadcasting division across each row
                return S  # Transpose to match the desired format
            
            def diag_divide(s, t):
                # return element-wise division of two vectors
                t[t == 0] = 1e-3
                return s / t

            projected_grad = loss1 - loss12 -loss2 + loss22
            

            g1 = torch.Generator(model.device).manual_seed(random_seed)
            g2 = torch.Generator(model.device).manual_seed(random_seed2)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=g1)
                z2 = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=g2)
                
                max_float16 = 65504
                
                # debuug code:
                # nan_indices = torch.where(torch.isnan(self.Hessian_matrix[name][0]))[0]
                # Hessian_estimator[0][nan_indices]
                
                # print(self.Hessian_matrix[name][0])

                # self.Hessian_matrix[name] = ((1-Hessian_smooth) * self.Hessian_matrix[name] +  Hessian_smooth * Hessian_estimator)
                # import pdb; pdb.set_trace()
                grad = (loss1-loss2)/(2 * self.args.zo_eps) * z 


                # t1 = diag_divide(grad, 2 *self.args.zo_eps * z)
                # t1 = diag_divide()
                t1 = projected_grad / (2 * self.args.zo_eps * self.args.zo_eps) 
                # t1 = diag_divide(projected_grad , (2 * self.args.zo_eps ** 2) * z * z2)
                t1.clamp_(min=-max_float16, max=max_float16)
                t1 = t1 * z
                # t1 = diag_divide(t1, z)
                t1.clamp_(min=-max_float16, max=max_float16)
                t1 = t1 * z2
                # t1 = diag_divide(t1, z2)
                t1.clamp_(min=-max_float16, max=max_float16)
                # t2 = t1.t()
                
                # self.Hessian_matrix = 1/2 * (t1 + t2)
                # self.Hessian_matrix = t1.abs()
                # self.Hessian_matrix = 1/ self.Hessian_matrix
                
                self.Hessian_matrix = diag_divide(1, t1.abs())
                print(self.Hessian_matrix)

                d = torch.sqrt(self.Hessian_matrix)
                d.clamp_(min=-max_float16, max=max_float16)
                grad *= d
                grad.clamp_(min=-max_float16, max=max_float16)
                
                param.data = param.data - zo_learning_rate * (grad + self.args.weight_decay * param.data)
                param.data.clamp_(min=-max_float16, max=max_float16)
                if param.data.isnan().any():
                    import pdb; pdb.set_trace()

                # if name == "model.decoder.layers[0].fc1.weight":
                    # print(1)

                # import pdb; pdb.set_trace()
            loss_out = self.zo_forward(model, inputs)

        return loss_out

    def myzo(self, model, inputs, zo_learning_rate, Hessian_smooth):
    
        # if self.Hessian_matrix is None:
        #     # self.zo_random_seed = np.random.randint(1000000000)
        #     self.Hessian_matrix = {}
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             self.Hessian_matrix[name] = torch.ones(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        random_seed = np.random.randint(1000000000)
        with torch.no_grad():
            loss_original = self.zo_forward(model, inputs)

            # first function evaluation
            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            self.zo_perturb_parameters(random_seed, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)


            # second function evaluation
            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=-2)
            self.zo_perturb_parameters(random_seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            class ComputeCovA:

                @classmethod
                def __init__(cls, projected_grad, zo_learning_rate, weight_decay):
                    cls.projected_grad = projected_grad
                    cls.zo_learning_rate = zo_learning_rate
                    cls.weight_decay = weight_decay

                @classmethod
                def __call__(cls, module, inputs):
                    # if isinstance(module, nn.Linear):
                    # print(module)
                    if inputs[0].isnan().any():
                        import pdb; pdb.set_trace()
                    if hasattr(module, "weight"):
                        # cov_a = cls.linear(module, inputs)
                        # print(inputs[0].shape)
                        # print(inputs[0])

                        if (not isinstance(module, nn.Linear)) or (torch.tensor(module.weight.data.size()).max() > 768):
                            z = torch.normal(mean=0, std=1, size=module.weight.data.size(), device=module.weight.data.device, dtype=module.weight.data.dtype)
                            grad = cls.projected_grad * z
                        else:
                            grad = cls.linear(module, inputs)[0]
                            if grad.isnan().any():
                                import pdb; pdb.set_trace()
                        module.weight.data -= cls.zo_learning_rate * (grad + cls.weight_decay * module.weight.data)
                        if module.weight.data.isnan().any():
                            import pdb;pdb.set_trace()
                        if hasattr(module, "bias"):
                            if module.bias is not None:
                                z = torch.normal(mean=0, std=1, size=module.bias.data.size(), device=module.bias.data.device, dtype=module.bias.data.dtype)
                                grad = cls.projected_grad * z
                                module.bias.data -= cls.zo_learning_rate * (grad + cls.weight_decay * module.bias.data)
                                if module.bias.data.isnan().any():
                                    import pdb;pdb.set_trace()
                    else:
                        # FIXME(CW): for extension to other layers.
                        # raise NotImplementedError
                        # cov_a = None
                        v = None
                        pass

                        # import pdb; pdb.set_trace()

                    # return cov_a

                @staticmethod
                def linear(layer, inputs):
                    # a: batch_size * in_dim

                    activation = inputs[0].view(-1, layer.weight.size(1)).to(torch.float32)
                    batch_size = activation.size(0)
                    # if layer.bias is not None:
                        # a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

                    # m_aa: [in_dim, in_dim]
                    m_aa = activation.t() @ (activation / batch_size)
                    max_float16 = 65504
                    # m_aa.clamp_(max=max_float16)
                    m_aa.clamp_(min=-max_float16, max=max_float16)
                    # d_a: [in_dim] 
                    # Q_a: [in_dim, in_dim]
                    # d_a, Q_a = torch.symeig(m_aa, eigenvectors=True)
                    try:
                        d_a, Q_a = torch.linalg.eigh(m_aa, UPLO='U')
                    except:
                        import pdb; pdb.set_trace()
                    d_a = d_a.to(inputs[0].dtype)
                    d_a.clamp_(min=-max_float16, max=max_float16)
                    Q_a = Q_a.to(inputs[0].dtype)
                    Q_a.clamp_(min=-max_float16, max=max_float16)

                    # weight: [out_dim, in_dim]
                    param = layer.weight

                    # z: [out_dim, in_dim]
                    # g(grad): [out_dim, in_dim]
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    g = self.projected_grad * z
                    # import pdb; pdb.set_trace()
                    # p_grad_mat = g.view(activation.size())
                    # p_grad_mat = g.view(-1,activation.size(1))
                    p_grad_mat = g
                    g = g.t()
            
                    m_gg = (g.t() @ (g / g.size(0))).to(torch.float32)
                    m_gg.clamp_(min=-max_float16, max=max_float16)
            
                    # d_g, Q_g = torch.symeig(m_gg, eigenvectors=True)
                    try:
                        d_g, Q_g = torch.linalg.eigh(m_gg, UPLO='U')
                    except:
                        import pdb; pdb.set_trace()
                    d_g = d_g.to(inputs[0].dtype)
                    d_g.clamp_(min=-max_float16, max=max_float16)
                    Q_g = Q_g.to(inputs[0].dtype)
                    Q_g.clamp_(min=-max_float16, max=max_float16)

                    
                    # Q_g: [input, input]
                    # p_grad_mat: [batch, input]

                    v1 = Q_g.t() @ p_grad_mat @ Q_a
                    damping = 0.001
                    tmp = (d_g.unsqueeze(1) * d_a.unsqueeze(0) + damping)
                    tmp.clamp_(min=-max_float16, max=max_float16)
                    # v2 = v1 / (d_g.unsqueeze(1) * d_a.unsqueeze(0) + damping)
                    v2 = v1/tmp
                    v2.clamp_(min = -max_float16, max=max_float16)
                    t1 = Q_g @ v2
                    t2 = v2 @ Q_a.t()
                    t2.clamp_(min = -max_float16, max=max_float16)
                    # v = Q_g @ v2 @ Q_a.t()
                    v = Q_g @ t2
                    v.clamp_(min = -max_float16, max=max_float16)
                    # v = [v.view(activation.data.size())]
                    v = [v.view(p_grad_mat.size())]
                    if v[0].isnan().any():
                        import pdb; pdb.set_trace()
                    return v

                # skip kl clip for now
                # vg_sum = 0
                # for m in self.modules:
                #     v = updates[m]
                #     vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
                #     if m.bias is not None:
                #         vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
                # nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))



            # import pdb; pdb.set_trace()
            projected_grad = (loss1-loss2)/(2 * self.args.zo_eps)
            self.projected_grad = projected_grad

            # debug
            # hooks = []
            # for module in self.model.modules():
            #     # classname = module.__class__.__name__
            #     # print('=> We keep following layers in KFAC. <=')
            #     # if classname in self.known_modules:
            #         # self.modules.append(module)
            #     # print(module)
            #     # print("###")
            #     if hasattr(module, 'weight'):
            #         handle = module.register_forward_pre_hook(ComputeCovA(projected_grad, zo_learning_rate, self.args.weight_decay))
            #         hooks.append(handle)
            #         # print('(%s): %s' % (count, module))
            #         # count += 1

            # model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            self.zo_perturb_parameters(random_seed, scaling_factor=1)
            # self.saved_model = copy.deepcopy(model)
            
            torch.manual_seed(random_seed)


            # for name, param in self.named_parameters_to_optim:
            #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            #     # Hessian_temp = (1/self.Hessian_matrix[name] * z * z)
            #     # Hessian_estimator = (torch.abs(loss1+loss2-2 * loss_original)* Hessian_temp  /(2 * self.args.zo_eps*self.args.zo_eps))
                
            #     # 设置上限值
            #     # 65504 exceeds the max value of float16
            #     # max_float16 = 65504
            #     # Hessian_estimator.clamp_(max=max_float16)

            #     # debuug code:
            #     # nan_indices = torch.where(torch.isnan(self.Hessian_matrix[name][0]))[0]
            #     # Hessian_estimator[0][nan_indices]
                
            #     # print(self.Hessian_matrix[name][0])

            #     # self.Hessian_matrix[name] = ((1-Hessian_smooth) * self.Hessian_matrix[name] +  Hessian_smooth * Hessian_estimator)

            #     grad = (loss1-loss2)/(2 * self.args.zo_eps) * z
            #     param.data = param.data - zo_learning_rate * (grad + self.args.weight_decay * param.data)

            #     # if name == "model.decoder.layers[0].fc1.weight":
            #         # print(1)

            #     # import pdb; pdb.set_trace()
            loss_out = self.zo_forward(model, inputs)

            # for handle in hooks:
            #     handle.remove()

        return loss_out
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
    
    def svrg_step(self, model, inputs, flag):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # Perturb running theta model

        # What parameters to optimize 

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)


        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        if flag == 1:
            self.saved_grad = self.projected_grad

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

    def svrg_update(self, model, flag = 1):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

            if flag == 0:
                param.data = param.data - self._get_learning_rate() * (self.saved_grad * z + args.weight_decay * param.data)

        self.lr_scheduler.step()

    def zo_vrpge_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        args.K = 2

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            # else:
            #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
            if "scores" in name:
                pass
            # if False:
                # param.data = param.data - 10 * self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                # param.data = param.data - 10 * 1/(args.K-1)*(self.fn_list[0] - self.fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(self.fn_list[1] - self.fn_avg)*getattr(m, 'stored_mask_1')
            elif "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        for n, m in model.named_modules():
            if hasattr(m, "scores"):
                # import pdb; pdb.set_trace()
                m.scores.data = m.scores.data - 50000 * self._get_learning_rate() * ( 1/(args.K-1)*(self.fn_list[0] - self.fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(self.fn_list[1] - self.fn_avg)*getattr(m, 'stored_mask_1'))
                # pass

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
