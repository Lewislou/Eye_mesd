# Eye-MESD pretraining

English. This repository releases the Eye-MESD group-relative policy optimization (GRPO) pretraining code and the backbone used for the paper's downstream evaluation. The weight file is the EMA teacher of a DINOv2 ViT-L/14 student at iteration 479999 (303,228,928 parameters, flat state dict). Teacher foundation models are not included. Training code derived from DINOv2 remains under the Apache License 2.0.

本仓库发布 Eye-MESD 的强化学习预训练代码，以及论文下游实验使用的 ViT-L/14 backbone。

## 仓库里有什么

| 路径 | 内容 |
| --- | --- |
| `dinov2/train/ssl_meta_arch.py` | GRPO 路由、三教师蒸馏、DINO / iBOT / KoLeo |
| `dinov2/train/train.py` | 训练循环、学习率与 EMA 调度 |
| `dinov2/fms.py` | EyeCLIP、RETFound、DINOv2-Giant 三个冻结教师 |
| `dinov2/kl_loss.py` | 蒸馏损失 |
| `configs/eye_mesd_pretrain.yaml` | 产出 iteration 479999 的那次训练配置，路径已改成占位符 |
| `checkpoints/Eye_mesd_pretrained.pth` | 发布权重，1.2 GB |
| `scripts/load_backbone.py` | 把发布权重载入 ViT-L/14 |
| `scripts/train_grpo.sh` | 启动预训练 |

训练代码来自 DINOv2，文件头保留 Meta 的 Apache-2.0 版权声明。Eye-MESD 改动集中在 `ssl_meta_arch.py`、`fms.py`、`kl_loss.py` 和 `dinov2/data/datasets/pathology.py`。

## 发布的权重

`checkpoints/Eye_mesd_pretrained.pth` 是 iteration **479999** 的 EMA teacher backbone。DINOv2 下游用的是 EMA teacher，不是正在更新的 online student。这份文件就是论文里全部 Eye-MESD 下游评测所用的编码器：343 个参数张量，303,228,928 个参数，没有优化器，也没有 FSDP 分片前缀。键名是 `cls_token`、`pos_embed`、`patch_embed.*`、`blocks.{0..23}.*`。

按 `OFFICIAL_EPOCH_LENGTH = 7500`，iteration 479999 是第 64 个 epoch 结束时的存档。配置本身训练 100 个 epoch。479999 是按下游验证选出来、并用于论文评测的那一次，不是最后一个 epoch。

SHA256 见 `CHECKSUMS.txt`：

```
28b1630da44f7870a65f36dcbb8255a5ff3f87da78075a40048d8b3e3fb7f7fe
```

训练时 FSDP 会把 student 按 `block_chunks: 4` 切块，完整训练状态大约 12 GB。发布文件在导出时已经展开成普通 backbone，加载时使用 `block_chunks=0`。

权重文件在磁盘上，`.gitignore` 忽略了 `*.pth`，避免把 1.2 GB 写进 git 历史。打包发布时把 `checkpoints/Eye_mesd_pretrained.pth` 一起拷走，或改用 Git LFS。

### 加载

在仓库根目录：

```bash
PYTHONPATH=. python scripts/load_backbone.py \
  --checkpoint checkpoints/Eye_mesd_pretrained.pth
```

脚本构建 `vit_large(patch_size=14, block_chunks=0)` 并 `load_state_dict`。已核对：343 个键全部对上，missing 与 unexpected 都是 0。`pos_embed` 形状是 `(1, 257, 1024)`，对应 224×224、patch 14。下游若改成 518，DINOv2 的 `interpolate_pos_encoding` 会按输入尺寸插值位置编码，不需要另存一份权重。

## 重新训练

教师权重和图像列表不在本仓库里，需要自行放到下面的路径，或改 `configs/eye_mesd_pretrain.yaml`。

| 配置项 | 期望文件 | 来源 |
| --- | --- | --- |
| `student.pretrained_weights` | `weights/dinov2_vitl14_pretrain.pth` | 官方 DINOv2 ViT-L/14 |
| `distill.eyeclip_weights` | `weights/eyeclip_visual.pt` | EyeCLIP visual 权重 |
| `distill.retfound_weights` | `weights/RETFound_dinov2.pth` | RETFound-DINOv2 |
| `distill.giant_weights` | `weights/dinov2_vitg14_reg4_pretrain.pth` | 官方 DINOv2 ViT-G/14 with registers |
| `train.dataset_path` | `data/images.json` | 预训练图像路径列表 |

`data/images.json` 可以是路径字符串的列表，也可以是「路径 → 标签」的字典。`PathologyDataset` 只读取路径，标签固定为 0。坏图会被替换成备用图，路径追加写入 `./failed_paths.txt`。

```bash
bash scripts/train_grpo.sh
```

默认 `NPROC=2`，每卡 `batch_size_per_gpu: 80`。单卡时：

```bash
NPROC=1 bash scripts/train_grpo.sh
```

配置里 `train.resume` 设成 `false`，方便从头训练。论文那次运行的存档目录里这个字段是 `true`，因为中途从已有 checkpoint 接着训。学生初始化权重是官方 `dinov2_vitl14_pretrain.pth`。

依赖见 `requirements.txt`。原训练环境是 PyTorch 2.1.2 与 xformers 0.0.23.post1。

## 超参数

下面的数字来自产出 479999 的配置，以及 `ssl_meta_arch.py` 里写死的 GRPO 常数。

| 符号或设置 | 数值 | 代码位置 |
| --- | --- | --- |
| λ1，DINO | 1.0 | `dino.loss_weight` |
| λ2，iBOT | 1.0 | `ibot.loss_weight` |
| λ3，KoLeo | 0.1 | `dino.koleo_loss_weight`，直接加到总损失上 |
| λ4，RL | 1.0 | `rl_router_loss` 无额外系数地加进 `loss_accumulator` |
| α | 0.1 | `compute_patch_reward` 里的 `alpha_sensitivity` |
| β | 0.05 | `beta_kl` |
| G | 4 | `RLRouter.forward(..., G=4)` |
| 组内优势的 ε | 1e-8 | `rewards.std + 1e-8` |
| 坍缩惩罚 | 局部方差 &lt; 1e-4 时为 −10，否则为 0 | `compute_patch_reward` |
| 蒸馏 cls / patch | 1.5 / 0.75 | 三个教师相同 |
| `DISTILL_SCALE` | 1.0 | 蒸馏项的额外倍率 |
| router 学习率倍率 | 0.2 | `optim.router_lr_multiplier` |
| GRPO 掩码比例 | Uniform(0.3, 0.8) | `ssl_meta_arch.py` 写死，不读 yaml |
| iBOT 掩码比例 | Uniform(0.1, 0.5) | `ibot.mask_ratio_min_max` |

奖励是 `r = −L_cons + α · variance + I_collapse`。`L_cons` 是被掩掉 patch 上的 1 − 余弦相似度。组内优势是 `(r − μ) / (σ + 1e-8)`。策略损失是 `−mean(log π · A)`，再加 `β · KL`。三个教师的奖励在这个版本里不做滑动标准化；组内标准化是唯一的奖励归一化。

优化器是 AdamW，β1 = 0.9，β2 = 0.999。`build_schedulers` 的学习率峰值是 `optim.base_lr = 2e-4`，warmup 从 0 升到 2e-4（10 个 epoch），再余弦降到 `min_lr = 1e-6`。权重衰减从 0.04 余弦升到 0.4。梯度裁剪 3.0。Layer-wise decay 0.95。EMA 动量从 0.992 升到 1.0。Teacher 温度从 0.04 升到 0.07，升温 30 个 epoch。全局 crop 224，局部 crop 98，局部 crop 数量 8。每个 epoch 7500 iter，共 100 epoch，每 4 个 epoch 存一次。

`apply_scaling_rules_to_cfg` 会按 `sqrt(global_batch / 1024)` 改写 `optim.lr` 并打印，训练循环不读取 `optim.lr`。配置里留下的 `5.59e-5` 是原运行在 global batch 80 时写入 yaml 的记录，等于 `2e-4 × sqrt(80 / 1024)`。实际步长是 `base_lr`。稿件里的全局 batch 160 对应 2 张 GPU × 每卡 80；这个脚本的默认就是这个设置。

## 许可

`LICENSE` 是 Apache License 2.0。DINOv2 源文件头部的 Meta 版权声明保持原样。EyeCLIP、RETFound、DINOv2-Giant 的权重不在本仓库中，使用时遵循各自的许可。

## 这个发布包里没有的东西

- 约 12 GB 的 FSDP 训练状态（含优化器与分片 student）
- 三个教师的权重，以及学生初始化用的官方 DINOv2 ViT-L/14
- 预训练图像与 `images.json`
- 分类、分割、VQA 的下游微调代码
- 早期试验脚本（滑动标准化版本、单教师试验、评测子目录）
