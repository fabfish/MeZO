# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

import contextlib
import math
import os
import random
import re
import shutil
import sys
import time
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from packaging import version

from transformers import Trainer, __version__
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
from transformers.dependency_versions_check import dep_version_check
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_scheduler
from transformers.trainer_callback import (
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    get_parameter_names,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    has_length,
    speed_metrics,
    ShardedDDPOption,
)
from transformers.utils import (
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

# 导入自定义 Scheduler
from lr_scheduler import zo_lr_scheduler
from Hessian_smooth_scheduler import Hessian_smooth_scheduler

# 环境设置
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
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)

# Checkpoint 文件名
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):
    """
    Custom Trainer implementing FOCUS (Forward-Only Coordinate Updates with Second-order Information).
    See paper: https://arxiv.org/abs/2305.17333 (MeZO base) and the attached FOCUS paper.
    """

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        # --- 1. 初始化训练步数与参数 ---
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = max(len_dataloader // args.gradient_accumulation_steps, 1)
        num_examples = self.num_examples(train_dataloader)

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)

        # --- 2. 优化器与调度器初始化 (Standard FO compatibility) ---
        # 如果是 ZO/FOCUS 方法，我们实际上不需要标准的 optimizer.step()，但为了兼容性保留
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        model = self._wrap_model(self.model)

        # 加载 Checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            # ... (省略复杂 checkpoint 加载细节以保持简洁，通常由 Trainer 基类处理)

        # --- 3. FOCUS (ZO-BCD) 初始化 ---
        # 自动识别 Transformer Layers 作为 Blocks
        self.layer_blocks = self._identify_blocks(model)
        self.num_blocks = len(self.layer_blocks)
        logger.info(f"Identified {self.num_blocks} blocks for BCD optimization.")
        
        # Bandit 算法初始化：每个 Block 被选中的概率 (均匀分布初始化)
        self.block_probs = np.ones(self.num_blocks) / self.num_blocks
        self.bandit_lr = 0.1  # Bandit 学习率，对应论文中的 alpha_p
        self.p_min = 0.01     # 最小探索概率

        # 状态追踪
        self.current_hessian_block_idx = None
        self.Hessian_matrix = {}  # 仅存储当前活跃 Block 的对角 Hessian
        self.zo_random_seed = np.random.randint(1000000000)

        # 开始训练循环
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        self.current_step = 0
        
        model.train()

        # 进度条
        epoch_iterator = train_dataloader

        for epoch in range(int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            for step, inputs in enumerate(epoch_iterator):
                self.current_step += 1
                
                # 准备输入
                inputs = self._prepare_inputs(inputs)

                # 动态学习率与 Hessian 平滑系数
                zo_lr = zo_lr_scheduler(
                    self.args.learning_rate, 
                    self.args.zo_lr_scheduler_type, 
                    self.args.warmup_step, 
                    self.args.decay_step, 
                    self.state.global_step, 
                    max_steps
                )
                hessian_smooth = Hessian_smooth_scheduler(
                    self.args.hessian_smooth_type, 
                    self.state.global_step, 
                    max_steps
                )

                # --- FOCUS 核心步骤 ---
                if args.trainer == "zo":
                    # 1. 选择活跃 Block (Bandit Selection)
                    # 论文建议每隔一定间隔更新 Block，或者是每次迭代更新。这里假设每次迭代更新。
                    # 实际优化：为了 Hessian 估计的稳定性，可以在同一个 Block 上训练几步再切换
                    active_block_idx = self._bandit_select_block()
                    
                    # 2. 切换 Hessian 缓存 (Memory Efficient)
                    # 如果选择了新的 Block，释放旧的显存并初始化新的 Hessian
                    self._manage_hessian_memory(model, active_block_idx)

                    # 3. 执行 FOCUS 更新 (ZO-BCD-Newton)
                    # 包括：扰动 -> Loss计算 -> 梯度估算 -> Hessian更新 -> 参数更新
                    loss_val, block_score = self.focus_step(
                        model, inputs, active_block_idx, zo_lr, hessian_smooth
                    )
                    
                    # 4. 更新 Bandit 概率 (Based on GSD Score)
                    self._update_bandit_probs(active_block_idx, block_score)

                    tr_loss_step = loss_val

                else:
                    # 标准 BP 训练 (Baseline)
                    tr_loss_step = self.training_step(model, inputs)
                    # 标准优化器步进
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()

                # 记录与回调
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / len(epoch_iterator)
                
                if self.state.global_step % args.logging_steps == 0:
                    log_metrics = {
                        "loss": tr_loss_step.item() if isinstance(tr_loss_step, torch.Tensor) else tr_loss_step,
                        "lr": zo_lr,
                        "step": self.state.global_step,
                        "active_block": active_block_idx if args.trainer == "zo" else -1,
                        "block_probs_max": np.max(self.block_probs) if args.trainer == "zo" else 0,
                        "block_probs_min": np.min(self.block_probs) if args.trainer == "zo" else 0,
                    }
                    self.log(log_metrics)

                # Evaluation & Saving
                if self.args.evaluation_strategy == "steps" and self.state.global_step % self.args.eval_steps == 0:
                    self.evaluate()
                    self.save_model()

                if self.state.global_step >= max_steps:
                    break
            
            if self.state.global_step >= max_steps:
                break

        logger.info("\n\nTraining completed.\n\n")
        return TrainOutput(self.state.global_step, 0.0, {})

    # --- Helper Functions for FOCUS ---

    def _identify_blocks(self, model):
        """
        自动识别 Transformer 的层作为 Blocks。
        返回一个 List[Dict]，每个 Dict 包含 {"name": str, "params": List[str]}
        """
        blocks = []
        
        # 1. 嵌入层 (Embedding)
        embed_params = []
        for name, param in model.named_parameters():
            # 适配常见的 embedding 命名
            if ("embed" in name or "wte" in name or "wpe" in name) and param.requires_grad:
                embed_params.append(name)
        if embed_params:
            blocks.append({"name": "embeddings", "params": embed_params})

        # 2. Transformer 层 (Layers)
        # 查找最大的层索引
        max_layer = -1
        for name, _ in model.named_parameters():
            # 匹配 layers.0.xxx 或 layer.0.xxx
            match = re.search(r'layers?\.(\d+)\.', name)
            if match:
                max_layer = max(max_layer, int(match.group(1)))
        
        if max_layer >= 0:
            for i in range(max_layer + 1):
                layer_params = []
                # 严格匹配第 i 层
                pattern = re.compile(f"layers?\.{i}\.")
                for name, param in model.named_parameters():
                    if pattern.search(name) and param.requires_grad:
                        layer_params.append(name)
                if layer_params:
                    blocks.append({"name": f"layer_{i}", "params": layer_params})

        # 3. 输出层 (Head/Norm)
        head_params = []
        for name, param in model.named_parameters():
            # 包含 lm_head, final_layer_norm, 且排除已经包含在 layers 中的
            is_layer = re.search(r'layers?\.(\d+)\.', name)
            if not is_layer and param.requires_grad and name not in embed_params:
                head_params.append(name)
        if head_params:
            blocks.append({"name": "head", "params": head_params})

        return blocks

    def _bandit_select_block(self):
        """
        根据当前的概率分布 p_t 采样一个 Block。
        对应论文 Algorithm 6 的采样部分。
        增加 NaN 检查，防止崩坏。
        """
        if np.any(np.isnan(self.block_probs)):
            logger.warning("Block probabilities contain NaN, resetting to uniform distribution.")
            self.block_probs = np.ones(self.num_blocks) / self.num_blocks
            
        return np.random.choice(len(self.layer_blocks), p=self.block_probs)

    def _manage_hessian_memory(self, model, active_block_idx):
        """
        关键内存优化：只在显存中保留当前 Active Block 的 Hessian。
        论文核心：Reduces memory requirements while accelerating convergence.
        """
        # 如果切换了 Block，释放旧的 Hessian
        if self.current_hessian_block_idx != active_block_idx:
            if hasattr(self, 'Hessian_matrix'):
                del self.Hessian_matrix
                torch.cuda.empty_cache() # 强制释放显存
            
            self.Hessian_matrix = {}
            self.current_hessian_block_idx = active_block_idx
            
            # 为新 Block 初始化 Hessian (单位矩阵/Ones)
            active_params = self.layer_blocks[active_block_idx]["params"]
            for name, param in model.named_parameters():
                if name in active_params:
                    # 使用与参数相同的 dtype 和 device
                    # 存储对角 Hessian (向量)
                    self.Hessian_matrix[name] = torch.ones_like(param.data)

    def _update_bandit_probs(self, block_idx, score):
        """
        更新 Bandit 概率分布 (Algorithm 6)。
        增加了数值稳定性检查。
        """
        # 检查 score 是否合法
        if np.isnan(score) or np.isinf(score):
            return # 如果 score 无效，跳过更新
        
        prob = self.block_probs[block_idx]
        # 避免除以零
        safe_prob = prob if prob > 1e-6 else 1e-6
        
        # 指数更新: p_new = p_old * exp(lr * estimated_reward)
        # estimated_reward = score / prob (Importance Sampling)
        estimated_reward = score / safe_prob
        
        # 限制 reward 范围，防止 exp 溢出
        # bandit_lr 默认为 0.1，如果 reward 过大，exp 会变成 inf
        estimated_reward = np.clip(estimated_reward, -1e4, 1e4)
        
        multiplier = np.exp(self.bandit_lr * estimated_reward)
        
        # 检查 multiplier 是否合法
        if np.isnan(multiplier) or np.isinf(multiplier):
            multiplier = 1.0

        # 更新权重
        new_weight = self.block_probs[block_idx] * multiplier
        
        # 临时更新用于归一化
        temp_probs = self.block_probs.copy()
        temp_probs[block_idx] = new_weight
        
        normalization_factor = np.sum(temp_probs)
        
        # 归一化检查
        if normalization_factor <= 1e-12 or np.isnan(normalization_factor) or np.isinf(normalization_factor):
            # 如果数值崩坏，重置为均匀分布
            self.block_probs = np.ones(self.num_blocks) / self.num_blocks
        else:
            self.block_probs = temp_probs / normalization_factor
        
        # Mixing 策略，保证最小探索概率 p_min
        self.block_probs = (1 - self.p_min) * self.block_probs + self.p_min / self.num_blocks
        
        # 最终安全检查
        if np.any(np.isnan(self.block_probs)):
             self.block_probs = np.ones(self.num_blocks) / self.num_blocks

    def focus_step(self, model, inputs, block_idx, lr, hessian_smooth):
        """
        FOCUS 算法的核心单步更新。
        """
        active_param_names = set(self.layer_blocks[block_idx]["params"])
        
        # 收集需要优化的参数对象
        named_params_to_optim = []
        for name, param in model.named_parameters():
            if name in active_param_names and param.requires_grad:
                named_params_to_optim.append((name, param))

        # 生成随机种子，用于 In-place 复现扰动 (Memory Efficient)
        self.zo_random_seed = np.random.randint(1000000000)
        
        # --- Forward 1: 原始 Loss (用于 Hessian 估算) ---
        # HiZOO 使用 3 次 Forward: f(w), f(w+z), f(w-z)
        
        with torch.no_grad():
            loss_original = self.zo_forward(model, inputs)

            # --- Forward 2: Pos Perturbation (+) ---
            self._zo_perturb_block(named_params_to_optim, scaling_factor=1.0)
            loss1 = self.zo_forward(model, inputs)

            # --- Forward 3: Neg Perturbation (-) ---
            # 从 +1 变到 -1，需要 -2 的 scaling (In-place)
            self._zo_perturb_block(named_params_to_optim, scaling_factor=-2.0)
            loss2 = self.zo_forward(model, inputs)

            # 恢复原始参数 (从 -1 变回 0，需要 +1)
            self._zo_perturb_block(named_params_to_optim, scaling_factor=1.0)

        # --- 计算更新 ---
        # 投影梯度 (Projected Gradient): (L(w+z) - L(w-z)) / (2 * eps)
        projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)
        
        # Hessian 估算 (Abs value of second derivative approx)
        # H * z * z ~ L(w+z) + L(w-z) - 2L(w)
        hessian_sample_val = torch.abs(loss1 + loss2 - 2 * loss_original) / (self.args.zo_eps ** 2)
        
        # 限制数值范围防止 FP16 溢出
        max_val = 65504 if self.args.fp16 else 1e8
        hessian_sample_val = torch.clamp(hessian_sample_val, max=max_val)

        # 计算 Block Saliency Score (用于 Bandit)
        # GSD Score ~ ||Gradient||^2 / Hessian
        # 这里简化使用投影梯度的绝对值
        if torch.isnan(projected_grad) or torch.isinf(projected_grad):
            block_score = 0.0
        else:
            block_score = projected_grad.abs().item()

        # --- 应用更新 (仅对 Active Block) ---
        # 重新生成相同的 z
        torch.manual_seed(self.zo_random_seed) 
        
        for name, param in named_params_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            # 1. 更新 Hessian (EMA)
            h_update = hessian_sample_val # 这是一个标量，基于 loss 差分
            
            current_h = self.Hessian_matrix[name]
            new_h = (1 - hessian_smooth) * current_h + hessian_smooth * h_update
            
            # 确保 H 正定且不过小
            new_h = torch.clamp(new_h, min=1e-8, max=max_val)
            # 检查 NaN
            if torch.isnan(new_h).any():
                # 如果出现 NaN，不更新 Hessian，保持原样或重置
                new_h = current_h 
            
            self.Hessian_matrix[name] = new_h
            
            # 2. 计算 Preconditioned Update
            # 更新量： (g^T z) * z / sqrt(H)
            preconditioner = torch.sqrt(new_h)
            update = (projected_grad / preconditioner) * z
            
            # 3. 更新参数
            # Weight decay
            if self.args.weight_decay > 0:
                update += self.args.weight_decay * param.data
            
            # 检查更新量是否合法
            if torch.isnan(update).any():
                continue # 跳过该参数的更新

            param.data -= lr * update

        return loss_original, block_score

    def _zo_perturb_block(self, named_params, scaling_factor):
        """
        仅扰动指定的 Block 参数 (In-place)。
        使用 Hessian 进行 Preconditioned Perturbation (如果需要)。
        HiZOO 论文中，扰动也是经过 Hessian 缩放的: z' = Sigma^(1/2) z
        这里如果 Hessian_matrix 存的是 H，则 Sigma ~ H^-1。
        所以扰动 z' = z / sqrt(H)。
        """
        torch.manual_seed(self.zo_random_seed)
        for name, param in named_params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            h = self.Hessian_matrix[name]
            # 避免除零
            h_safe = torch.clamp(h, min=1e-8)
            
            # 扰动量 = scaling * eps * z * Sigma^(1/2)
            # Sigma = H^-1 => Sigma^(1/2) = 1/sqrt(H)
            perturbation = scaling_factor * self.args.zo_eps * z / torch.sqrt(h_safe)
            
            # 检查 NaN
            if torch.isnan(perturbation).any():
                 continue

            param.data += perturbation

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model.
        """
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
        """
        针对不可微目标（如 SQuAD F1）的前向传播。
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], 
                do_sample=args.sampling, 
                temperature=args.temperature, 
                num_beams=args.num_beams, 
                top_p=args.top_p, 
                top_k=args.top_k, 
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, 
                eos_token_id=self.tokenizer.eos_token_id,
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            
            # 简单的 F1 计算 (需确保 metrics.py 中的 f1 函数可用)
            from metrics import f1
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        # 修复 FSDP 保存问题的逻辑保持不变
        if output_dir is None:
            output_dir = self.args.output_dir
        
        if self.args.should_save:
            self._save(output_dir)