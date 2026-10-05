# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math  
import torch
from torch import nn
import torch.nn.functional as F
import time
from torch.distributions import Categorical

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model
from dinov2.models.vision_transformer import BlockChunk
from dinov2.fms import EyeCLIP, RETFound, DINOv2Giant  
# 确保你的 kl_loss.py 已经按照我们之前讨论的修改好了 (支持 reduction='none')
from dinov2.kl_loss import get_kd_loss, FeatureLoss

import copy 

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")

logger = logging.getLogger("dinov2")

# ==========================================
# 0. 核心工具函数：物理遮挡与奖励计算
# ==========================================
def apply_batch_pixel_mask_with_mask(img_tensor, mask_ratio, patch_size=16):
    B, C, H, W = img_tensor.shape
    p = patch_size
    h_grid, w_grid = H // p, W // p
    N = h_grid * w_grid

    x = img_tensor.reshape(B, C, h_grid, p, w_grid, p)
    x = torch.einsum('bchpwq->bhwpqc', x)
    patches = x.reshape(B, N, p**2 * C)

    noise = torch.rand(B, N, device=img_tensor.device)
    len_keep = int(N * (1 - mask_ratio))

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones([B, N], device=img_tensor.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    masked_patches = patches.clone()
    masked_patches[mask == 1] = 0

    x_masked = masked_patches.reshape(B, h_grid, w_grid, p, p, C)
    x_masked = torch.einsum('bhwpqc->bchpwq', x_masked)
    masked_img = x_masked.reshape(B, C, H, W)

    return masked_img, mask

def compute_patch_reward(encoder, imgs, mask_ratio):
    with torch.no_grad():
        feat_orig = encoder(imgs)['patch_token']   
        
        B, C, H, W = imgs.shape
        N = feat_orig.shape[1]  
        actual_patch_size = H // int(math.sqrt(N))  
        
        masked_imgs, mask = apply_batch_pixel_mask_with_mask(
            imgs, mask_ratio, patch_size=actual_patch_size
        )
        
        feat_masked = encoder(masked_imgs)['patch_token']  
        
        mask_bool = mask.bool() 
        if mask_bool.sum() == 0: return None  

        feat_masked_norm = F.normalize(feat_masked, dim=-1)
        feat_orig_norm = F.normalize(feat_orig, dim=-1)
        
        cos_sim = F.cosine_similarity(feat_masked_norm, feat_orig_norm, dim=-1) 
        masked_cos_sim = cos_sim * mask 
        num_masked_per_img = mask.sum(dim=1) 
        
        per_img_consistency = masked_cos_sim.sum(dim=1) / (num_masked_per_img + 1e-6)
        per_img_loss = 1.0 - per_img_consistency 

        pred_flat = feat_masked[mask_bool] 
        pred_normalized = F.normalize(pred_flat, p=2, dim=-1)
        mean_sensitivity = torch.var(pred_normalized, dim=0).mean() 
        
        collapse_penalty = -10.0 if mean_sensitivity < 1e-4 else 0.0
        alpha_sensitivity = 0.1 

        reward = -per_img_loss + (alpha_sensitivity * mean_sensitivity) + collapse_penalty

    return reward

# ==========================================
# 1. 强化学习路由器 (GRPO 采样能力)
# ==========================================
class RLRouter(nn.Module):
    def __init__(self, state_dim=1024, num_experts=3):
        super().__init__()
        self.brain = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        
    def forward(self, state, G=4):
        logits = self.brain(state)
        m = Categorical(logits=logits)
        
        if self.training:
            raw_actions = m.sample((G,))  
            raw_log_probs = m.log_prob(raw_actions)  
            
            actions = raw_actions.transpose(0, 1) 
            log_probs = raw_log_probs.transpose(0, 1) 
            return actions, log_probs, logits
        else:
            actions = torch.argmax(logits, dim=-1).unsqueeze(1) 
            return actions, None, logits

# ==========================================
# 3. 核心元架构 (SSLMetaArch)
# ==========================================
class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()
        frozen_teachers_dict = dict()
        
        self.use_eyeclip_flag = True if getattr(cfg.distill, "EyeCLIP_loss_ratio", 0) > 0 else False
        self.use_retfound_flag = True if getattr(cfg.distill, "RETFound_loss_ratio", 0) > 0 else False
        self.use_giant_flag = True if getattr(cfg.distill, "Giant_loss_ratio", 0) > 0 else False

        if self.use_eyeclip_flag:
            frozen_teachers_dict['eyeclip'] = EyeCLIP(weight_path=str(cfg.distill.eyeclip_weights))
        if self.use_retfound_flag:
            frozen_teachers_dict['retfound'] = RETFound(weight_path=str(cfg.distill.retfound_weights))
        if self.use_giant_flag:
            frozen_teachers_dict['giant'] = DINOv2Giant(weight_path=str(cfg.distill.giant_weights))

        # 🌟 关键：使用 reduction='none' 获取每张图的独立 Loss
        self.distill_loss_fn = get_kd_loss(cfg.distill.loss_name, reduction='none')
        self.distill_feature_loss_fn = FeatureLoss(dim=-1, reduction='none')

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights, map_location="cpu")
            if 'teacher' in chkpt:
                chkpt = chkpt['teacher']
                chkpt = {k.replace("backbone.", ""): v for k, v in chkpt.items()}
            if "pos_embed" in chkpt:
                pos_embed_checkpoint = chkpt["pos_embed"]
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_extra_tokens = 1 
                if pos_embed_checkpoint.shape != student_backbone.pos_embed.shape:
                    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                    new_size = int((student_backbone.pos_embed.shape[-2] - num_extra_tokens) ** 0.5)
                    if orig_size != new_size:
                        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                        chkpt["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)
            student_backbone.load_state_dict(chkpt, strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes
        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        distill_heads = {}
        if self.use_eyeclip_flag:
            distill_heads['eyeclip'] = nn.Linear(embed_dim, frozen_teachers_dict['eyeclip'].dim)
            nn.init.orthogonal_(distill_heads['eyeclip'].weight); nn.init.zeros_(distill_heads['eyeclip'].bias)
        if self.use_retfound_flag:
            distill_heads['retfound'] = nn.Linear(embed_dim, frozen_teachers_dict['retfound'].dim)
            nn.init.orthogonal_(distill_heads['retfound'].weight); nn.init.zeros_(distill_heads['retfound'].bias)
        if self.use_giant_flag:
            distill_heads['giant'] = nn.Linear(embed_dim, frozen_teachers_dict['giant'].dim)
            nn.init.orthogonal_(distill_heads['giant'].weight); nn.init.zeros_(distill_heads['giant'].bias)
        self.distill_heads = nn.ModuleDict(distill_heads)

        self.num_experts = sum([self.use_eyeclip_flag, self.use_retfound_flag, self.use_giant_flag])
        
        if self.num_experts > 1:
            self.rl_router = RLRouter(state_dim=embed_dim, num_experts=self.num_experts)
            student_model_dict["rl_router"] = self.rl_router 
            
            self.ema_rl_router = copy.deepcopy(self.rl_router)
            for param in self.ema_rl_router.parameters(): param.requires_grad = False
            
            self.register_buffer("expert_reward_mean", torch.zeros(self.num_experts))
            self.register_buffer("expert_reward_var", torch.ones(self.num_experts))

        if self.do_dino or self.do_ibot:
            dino_head_args = dict(in_dim=embed_dim, out_dim=cfg.dino.head_n_prototypes, hidden_dim=cfg.dino.head_hidden_dim, bottleneck_dim=cfg.dino.head_bottleneck_dim, nlayers=cfg.dino.head_nlayers)
            self.dino_loss = DINOLoss(self.dino_out_dim)
            student_model_dict["dino_head"] = DINOHead(**dino_head_args)
            teacher_model_dict["dino_head"] = DINOHead(**dino_head_args)
            if self.do_koleo: self.koleo_loss = KoLeoLoss()

        if self.do_ibot:
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                student_model_dict["ibot_head"] = DINOHead(**dino_head_args)
                teacher_model_dict["ibot_head"] = DINOHead(**dino_head_args)

        self.need_to_synchronize_fsdp_streams = True
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.distillation = nn.ModuleDict(frozen_teachers_dict)
        
        for p in self.teacher.parameters(): p.requires_grad = False
        for p in self.distillation.parameters(): p.requires_grad = False

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None: self.fp16_scaler.scale(loss).backward()
        else: loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        ibot_loss_scale = 1.0 / n_global_crops

        loss_accumulator = 0
        loss_dict = {}

        # ==========================================
# [Step 1] 原生 Teacher Forward
        # ==========================================
        @torch.no_grad()
        def get_teacher_output():
            out_dict = self.teacher.backbone(global_crops, is_training=True)
            t_cls = out_dict["x_norm_clstoken"].chunk(n_global_crops)
            t_cls = torch.cat((t_cls[1], t_cls[0]))
            t_patch = out_dict["x_norm_patchtokens"]
            _dim = t_patch.shape[-1]
            n_cls = t_cls.shape[0]

            if self.do_ibot and not self.ibot_separate_head:
                buf = t_patch.new_zeros(upperbound + n_cls, _dim)
                buf[:n_cls].copy_(t_cls)
                torch.index_select(t_patch.flatten(0, 1), dim=0, index=mask_indices_list, out=buf[n_cls : n_cls + n_masked_patches])
                t_after_head = self.teacher.dino_head(buf)
                t_cls_out = t_after_head[:n_cls]
                t_patch_out = t_after_head[n_cls : n_cls + n_masked_patches]
            elif self.do_ibot and self.ibot_separate_head:
                buf = t_patch.new_zeros(upperbound, _dim)
                torch.index_select(t_patch.flatten(0, 1), dim=0, index=mask_indices_list, out=buf[:n_masked_patches])
                t_cls_out = self.teacher.dino_head(t_cls)
                t_patch_out = self.teacher.ibot_head(buf)[:n_masked_patches]
            else:
                t_cls_out = self.teacher.dino_head(t_cls)
                t_patch_out = None

            if self.cfg.train.centering == "centering":
                t_dino_soft = self.dino_loss.softmax_center_teacher(t_cls_out, teacher_temp=teacher_temp).view(n_global_crops, -1, *t_cls_out.shape[1:])
                self.dino_loss.update_center(t_cls_out)
                t_ibot_soft = self.ibot_patch_loss.softmax_center_teacher(t_patch_out.unsqueeze(0), teacher_temp=teacher_temp).squeeze(0) if self.do_ibot else None
                if self.do_ibot: self.ibot_patch_loss.update_center(t_patch_out[:n_masked_patches])
            else:
                t_dino_soft = self.dino_loss.sinkhorn_knopp_teacher(t_cls_out, teacher_temp=teacher_temp).view(n_global_crops, -1, *t_cls_out.shape[1:])
                t_ibot_soft = self.ibot_patch_loss.sinkhorn_knopp_teacher(t_patch_out, teacher_temp=teacher_temp, n_masked_patches_tensor=n_masked_patches_tensor) if self.do_ibot else None
            return t_dino_soft, t_ibot_soft

        teacher_dino_out, teacher_ibot_out = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        # ==========================================
# [Step 2 & 3] 多专家特征提取 & Student Forward
        # ==========================================
        @torch.no_grad()
        def get_distill_teacher_output():
            res = {}
            if self.use_eyeclip_flag: res['eyeclip'] = self.distillation.eyeclip(global_crops)
            if self.use_retfound_flag: res['retfound'] = self.distillation.retfound(global_crops)
            if self.use_giant_flag: res['giant'] = self.distillation.giant(global_crops)
            return res
        distill_teachers_out = get_distill_teacher_output()

        student_global_out, student_local_out = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )
        student_global_cls = student_global_out["x_norm_clstoken"]
        student_global_patch = student_global_out["x_norm_patchtokens"]

        DISTILL_SCALE = 1.0
        distill_weights = None

        # ==========================================
        # [Step 4] 逐专家蒸馏损失（单教师 / 多教师共用）
        # ==========================================
        expert_losses = {}

        if self.use_eyeclip_flag:
            s_cls = self.distill_heads.eyeclip(student_global_cls.unsqueeze(1))
            t_cls = distill_teachers_out['eyeclip']['cls_token'].detach().clone()
            raw_cls_loss = self.distill_loss_fn(s_cls, t_cls).view(-1)
            total_eyeclip_loss = raw_cls_loss * self.cfg.distill.EyeCLIP_loss_ratio
            if getattr(self.cfg.distill, "EyeCLIP_patch_ratio", 0) > 0:
                s_patch = self.distill_heads.eyeclip(student_global_patch)
                t_patch = distill_teachers_out['eyeclip']['patch_token'].detach().clone()
                raw_patch_loss = self.distill_feature_loss_fn(s_patch, t_patch)
                if raw_patch_loss.dim() > 1:
                    raw_patch_loss = raw_patch_loss.mean(dim=1)
                total_eyeclip_loss = total_eyeclip_loss + (raw_patch_loss.view(-1) * self.cfg.distill.EyeCLIP_patch_ratio)
            expert_losses[0] = total_eyeclip_loss
            loss_dict['distill_eyeclip_loss'] = total_eyeclip_loss.mean()

        if self.use_retfound_flag:
            s_cls = self.distill_heads.retfound(student_global_cls.unsqueeze(1))
            t_cls = distill_teachers_out['retfound']['cls_token'].detach().clone()
            raw_cls_loss = self.distill_loss_fn(s_cls, t_cls).view(-1)
            total_retfound_loss = raw_cls_loss * self.cfg.distill.RETFound_loss_ratio
            if getattr(self.cfg.distill, "RETFound_patch_ratio", 0) > 0:
                s_patch = self.distill_heads.retfound(student_global_patch)
                t_patch = distill_teachers_out['retfound']['patch_token'].detach().clone()
                raw_patch_loss = self.distill_feature_loss_fn(s_patch, t_patch)
                if raw_patch_loss.dim() > 1:
                    raw_patch_loss = raw_patch_loss.mean(dim=1)
                total_retfound_loss = total_retfound_loss + (raw_patch_loss.view(-1) * self.cfg.distill.RETFound_patch_ratio)
            expert_losses[1] = total_retfound_loss
            loss_dict['distill_retfound_loss'] = total_retfound_loss.mean()

        if self.use_giant_flag:
            s_cls = self.distill_heads.giant(student_global_cls.unsqueeze(1))
            t_cls = distill_teachers_out['giant']['cls_token'].detach().clone()
            raw_cls_loss = self.distill_loss_fn(s_cls, t_cls).view(-1)
            total_giant_loss = raw_cls_loss * getattr(self.cfg.distill, "Giant_loss_ratio", 1.0)
            if getattr(self.cfg.distill, "Giant_patch_ratio", 0) > 0:
                s_patch = self.distill_heads.giant(student_global_patch)
                t_patch = distill_teachers_out['giant']['patch_token'].detach().clone()
                raw_patch_loss = self.distill_feature_loss_fn(s_patch, t_patch)
                if raw_patch_loss.dim() > 1:
                    raw_patch_loss = raw_patch_loss.mean(dim=1)
                total_giant_loss = total_giant_loss + (raw_patch_loss.view(-1) * self.cfg.distill.Giant_patch_ratio)
            expert_losses[2] = total_giant_loss
            loss_dict['distill_giant_loss'] = total_giant_loss.mean()

        # ==========================================
        # [Step 5] 路由：多教师走 GRPO；单教师权重恒为 1
        # ==========================================
        if self.num_experts > 1:
            G = 4
            with torch.no_grad():
                state_B = self.teacher.backbone(global_crops, is_training=False).chunk(2, dim=0)[0].detach()

            actions, log_probs, curr_logits = self.student.rl_router(state_B, G=G)

            with torch.no_grad():
                expert_rewards = {}
                expert_names_map = {0: 'eyeclip', 1: 'retfound', 2: 'giant'}
                current_mask_ratio = torch.empty(1).uniform_(0.3, 0.8).item()
                imgs = global_crops.chunk(2, dim=0)[0]

                for i in range(self.num_experts):
                    expert_name = expert_names_map[i]
                    if getattr(self, f"use_{expert_name}_flag", False):
                        encoder = self.distillation[expert_name]
                        raw_reward = compute_patch_reward(encoder, imgs, current_mask_ratio)
                        if raw_reward is not None:
                            expert_rewards[i] = raw_reward

                B_size = actions.shape[0]
                rewards = torch.zeros(B_size, G, device=actions.device, dtype=torch.float32)
                for i in range(self.num_experts):
                    if i in expert_rewards:
                        mask_i = (actions == i)
                        rewards[mask_i] = expert_rewards[i].unsqueeze(1).expand(B_size, G)[mask_i]

                mean_reward = rewards.mean(dim=1, keepdim=True)
                std_reward = rewards.std(dim=1, keepdim=True) + 1e-8
                advantages = (rewards - mean_reward) / std_reward
                _, _, ref_logits = self.ema_rl_router(state_B, G=1)

            ref_probs = F.softmax(ref_logits, dim=-1)
            kl_div = F.kl_div(F.log_softmax(curr_logits, dim=-1), ref_probs, reduction='batchmean')
            policy_loss = -(log_probs * advantages.detach()).mean()
            beta_kl = 0.05
            rl_router_loss = policy_loss + beta_kl * kl_div

            loss_dict['rl_policy_loss'] = policy_loss
            loss_dict['rl_kl_div'] = kl_div
            loss_dict['rl_router_loss'] = rl_router_loss
            loss_dict['rl_mask_ratio'] = torch.tensor(current_mask_ratio, device=state_B.device)
            loss_accumulator += rl_router_loss

            distill_weights = F.softmax(curr_logits, dim=-1).detach()
            distill_weights_2B = distill_weights.repeat(2, 1)

            weighted_student_expert_loss = 0.0
            for i in range(self.num_experts):
                if i in expert_losses:
                    dynamic_weight_i = distill_weights_2B[:, i].view(-1)
                    weighted_student_expert_loss = weighted_student_expert_loss + (
                        dynamic_weight_i * expert_losses[i] * DISTILL_SCALE
                    )
            final_expert_loss = weighted_student_expert_loss.mean() if isinstance(weighted_student_expert_loss, torch.Tensor) else 0.0
            loss_dict['distill_soft_expert_loss'] = final_expert_loss
            loss_accumulator += final_expert_loss

        elif self.num_experts == 1 and expert_losses:
            final_expert_loss = next(iter(expert_losses.values())).mean() * DISTILL_SCALE
            loss_dict['distill_soft_expert_loss'] = final_expert_loss
            loss_accumulator += final_expert_loss

        # ==========================================
# [Step 6] 原生 DINO / iBOT Heads Loss
        # ==========================================
        inputs_for_head = [student_local_out["x_norm_clstoken"].unsqueeze(0), student_global_cls.unsqueeze(0)]
        if self.do_ibot:
            _dim = student_global_cls.shape[-1]
            buffer_patch = student_global_patch.new_zeros(upperbound, _dim)
            buffer_patch[:n_masked_patches].copy_(torch.index_select(student_global_patch.flatten(0, 1), dim=0, index=mask_indices_list))
            if not self.ibot_separate_head: inputs_for_head.append(buffer_patch.unsqueeze(0))
            else: student_global_masked_patch_head = self.student.ibot_head(buffer_patch)[:n_masked_patches]

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_head)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        
        student_local_cls_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_head = outputs_list.pop(0).squeeze(0)
        if self.do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            l_dino_local = self.dino_loss(student_local_cls_head.chunk(n_local_crops), teacher_dino_out) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_local_crops_loss"] = l_dino_local
            loss_accumulator += self.cfg.dino.loss_weight * l_dino_local

        if self.do_dino:
            l_dino_global = self.dino_loss([student_global_cls_head], [teacher_dino_out.flatten(0, 1)]) * 2 / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_global_crops_loss"] = l_dino_global
            loss_accumulator += self.cfg.dino.loss_weight * l_dino_global
            if self.do_koleo:
                l_koleo = self.cfg.dino.koleo_loss_weight * sum(self.koleo_loss(p) for p in student_global_cls.chunk(2))
                loss_accumulator += l_koleo
                loss_dict["koleo_loss"] = l_koleo / 2

        if self.do_ibot:
            l_ibot = self.ibot_patch_loss.forward_masked(student_global_masked_patch_head, teacher_ibot_out, student_masks_flat=masks, n_masked_patches=n_masked_patches, masks_weight=masks_weight) * 2 * ibot_loss_scale
            loss_dict["ibot_loss"] = l_ibot / 2
            loss_accumulator += self.cfg.ibot.loss_weight * l_ibot

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        if self.num_experts > 1 and distill_weights is not None:
            if self.use_eyeclip_flag:
                loss_dict["weight_expert_eyeclip"] = distill_weights[:, 0].mean()
            if self.use_retfound_flag:
                loss_dict["weight_expert_retfound"] = distill_weights[:, 1].mean()
            if self.use_giant_flag:
                loss_dict["weight_expert_giant"] = distill_weights[:, 2].mean()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                if hasattr(self.teacher.backbone, attr):
                    stream = getattr(self.teacher.backbone, attr)
                    setattr(self.student.dino_head, attr, stream)
                    setattr(self.teacher.dino_head, attr, stream)
                    setattr(self.student.backbone, attr, stream)
                    if self.use_eyeclip_flag: setattr(self.distill_heads.eyeclip, attr, stream)
                    if self.use_retfound_flag: setattr(self.distill_heads.retfound, attr, stream)
                    if self.use_giant_flag: setattr(self.distill_heads.giant, attr, stream)
                    if hasattr(self.student, "rl_router"): setattr(self.student.rl_router, attr, stream)
                    if hasattr(self, "ema_rl_router"): setattr(self.ema_rl_router, attr, stream)
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        with torch.no_grad():
            for k in self.student.keys():
                if k == "rl_router": continue
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    torch._foreach_mul_(mt.params, m)
                    torch._foreach_add_(mt.params, ms.params, alpha=1 - m)
            
            if hasattr(self, "ema_rl_router"):
                for ms, mt in zip(self.student.rl_router.parameters(), self.ema_rl_router.parameters()):
                    mt.data.mul_(m).add_(ms.data, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()
        self.distillation.eval()
        if hasattr(self, "ema_rl_router"): self.ema_rl_router.eval()

    def get_params_groups(self):
        all_params_groups = []
        
        for name, m in self.student.items():
            fused_groups = self.get_maybe_fused_params_for_submodel(m)
            
            if name == "rl_router" and hasattr(self.cfg.optim, "router_lr_multiplier"):
                for g in fused_groups:
                    g["lr_multiplier"] = self.cfg.optim.router_lr_multiplier
                    
            all_params_groups += fused_groups
            
        for m in self.distill_heads.values(): 
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
            
        return all_params_groups

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(model=m, lr_decay_rate=self.cfg.optim.layerwise_decay, patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult)
        fused_params_groups = fuse_params_groups(params_groups)
        for g in fused_params_groups: g["foreach"] = True
        return fused_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- Preparing Models & RL Router")
        for k, v in self.student.items():
            if k in self.teacher:
                self.teacher[k].load_state_dict(self.student[k].state_dict())
                self.teacher[k] = get_fsdp_wrapper(self.cfg.compute_precision.teacher[k], modules_to_wrap={BlockChunk})(self.teacher[k])
            if k == "rl_router":
                self.student[k] = get_fsdp_wrapper(self.cfg.compute_precision.student.backbone, modules_to_wrap=set())(self.student[k])
                self.ema_rl_router = get_fsdp_wrapper(self.cfg.compute_precision.student.backbone, modules_to_wrap=set())(self.ema_rl_router)
            else:
                self.student[k] = get_fsdp_wrapper(self.cfg.compute_precision.student[k], modules_to_wrap={BlockChunk})(self.student[k])

        for k in self.distill_heads.keys():
            head_cfg = getattr(self.cfg.compute_precision, "distill_heads", self.cfg.compute_precision.student.backbone)
            if isinstance(head_cfg, dict) and k in head_cfg: head_cfg = head_cfg[k]
            self.distill_heads[k] = get_fsdp_wrapper(head_cfg, modules_to_wrap=set())(self.distill_heads[k])
        
        for k in self.distillation.keys():
            t_cfg = self.cfg.compute_precision.teacher.backbone
            self.distillation[k] = get_fsdp_wrapper(t_cfg, modules_to_wrap={BlockChunk})(self.distillation[k])