import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from PIL import Image                   # 新增：用于加载真实图像
from torchvision import transforms      # 新增：用于图像预处理

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("eye_mesd")

# ==========================================
# 1. EyeCLIP Adapter
# ==========================================
class EyeCLIP(nn.Module):
    def __init__(self, weight_path='weights/eyeclip_visual.pt') -> None:
        super().__init__()
        self.dim = 768
        self.model = timm.create_model("vit_base_patch32_224", pretrained=False, num_classes=0)
        
        if os.path.exists(weight_path):
            self._load_weights(weight_path)
        else:
            logger.warning(f"EyeCLIP weights not found at {weight_path}, using random init.")

        for param in self.model.parameters(): param.requires_grad = False

    def _load_weights(self, weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'visual' not in k or 'decoder' in k: continue
                nk = k.replace('visual.', '').replace('transformer.resblocks.', 'blocks.')
                nk = nk.replace('ln_1.', 'norm1.').replace('ln_2.', 'norm2.')
                nk = nk.replace('c_fc.', 'fc1.').replace('c_proj.', 'fc2.')
                
                if 'attn.in_proj_weight' in nk:
                    new_state_dict[nk.replace('attn.in_proj_weight', 'attn.qkv.weight')] = v; continue
                if 'attn.in_proj_bias' in nk:
                    new_state_dict[nk.replace('attn.in_proj_bias', 'attn.qkv.bias')] = v; continue
                
                nk = nk.replace('attn.out_proj.', 'attn.proj.').replace('conv1.weight', 'patch_embed.proj.weight')
                nk = nk.replace('ln_pre.', 'norm_pre.').replace('ln_post.', 'norm.')
                
                if 'class_embedding' in nk:
                    if v.ndim == 1: v = v.reshape(1, 1, -1)
                    new_state_dict['cls_token'] = v; continue
                if 'positional_embedding' in nk:
                    if v.ndim == 2: v = v.unsqueeze(0)
                    new_state_dict['pos_embed'] = v; continue
                new_state_dict[nk] = v
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"EyeCLIP Loaded. Missing: {len(msg.missing_keys)}")
        except Exception as e: logger.error(f"Error loading EyeCLIP: {e}"); raise e

    def forward(self, img):
        f = self.model.forward_features(img)
        return {'cls_token': f[:, :1, :], 'patch_token': f[:, 1:, :]}

# ==========================================
# 2. RETFound Adapter
# ==========================================
class RETFound(nn.Module):
    def __init__(self, weight_path='weights/RETFound_dinov2.pth') -> None:
        super().__init__()
        self.dim = 1024
        self.model = timm.create_model("vit_large_patch14_dinov2", img_size=518, patch_size=14, num_classes=0, dynamic_img_size=True)
        if os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
                sd = checkpoint.get('teacher', checkpoint.get('model', checkpoint))
                self.model.load_state_dict({k.replace('backbone.', ''): v for k, v in sd.items()}, strict=False)
                logger.info(f"RETFound Loaded.")
            except Exception as e: logger.error(f"Error loading RETFound: {e}")
        else: logger.warning(f"RETFound not found at {weight_path}")
        for param in self.model.parameters(): param.requires_grad = False
    
    def forward(self, img):
        f = self.model.forward_features(img)
        return {'cls_token': f[:, :1, :], 'patch_token': f[:, 1:, :]}

# ==========================================
# 3. DINOv2 Giant (Reg4) - 最终修正版
# ==========================================
class DINOv2Giant(nn.Module):
    def __init__(self, weight_path='weights/dinov2_vitg14_reg4_pretrain.pth') -> None:
        super().__init__()
        self.dim = 1536
        # 创建 Giant 模型
        self.model = timm.create_model("vit_giant_patch14_reg4_dinov2", img_size=518, patch_size=14, num_classes=0, dynamic_img_size=True)
        
        if os.path.exists(weight_path): 
            self._load_weights(weight_path)
        else: 
            logger.warning(f"DINOv2 Giant weights not found at {weight_path}")
            
        for param in self.model.parameters(): param.requires_grad = False

    def _load_weights(self, weight_path):
        try:
            logger.info(f"Loading Giant weights from {weight_path}...")
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get('model', checkpoint)
            
            new_state_dict = {}

            for k, v in state_dict.items():
                # 1. 移除 backbone 前缀 (如果有)
                nk = k.replace('backbone.', '')
                
                # 2. Register Token 映射
                if 'register_tokens' in nk: nk = nk.replace('register_tokens', 'reg_token')
                
                # 3. Pos Embed 切片 (1370 -> 1369)
                if 'pos_embed' in nk and v.shape[1] == self.model.pos_embed.shape[1] + 1:
                    v = v[:, 1:, :]

                # 4. 关键修正：直接映射 w12 -> fc1, w3 -> fc2
                if 'mlp.w12.' in nk:
                    nk = nk.replace('mlp.w12.', 'mlp.fc1.')
                elif 'mlp.w3.' in nk:
                    nk = nk.replace('mlp.w3.', 'mlp.fc2.')

                new_state_dict[nk] = v

            # 加载权重
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            
            if len(msg.missing_keys) == 0:
                logger.info("✅ DINOv2 Giant Loaded Perfectly (0 missing keys).")
            else:
                logger.warning(f"⚠️  Loaded with missing keys: {len(msg.missing_keys)}")
                logger.warning(f"Sample missing: {msg.missing_keys[:5]}")

        except Exception as e:
            logger.error(f"Error loading DINOv2 Giant: {e}")
            raise e

    def forward(self, img):
        features = self.model.forward_features(img)
        # [CLS, Reg*4, Patches...]
        return {'cls_token': features[:, :1, :], 'patch_token': features[:, 5:, :]}


# ==========================================
# 4. 测试函数与指标计算模块
# ==========================================
def generate_masked_image(img, mask_ratio=0.5, patch_size=14):
    """
    模拟 Patch 级别的随机掩码
    """
    B, C, H, W = img.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # 生成随机掩码 (0 为保留, 1 为遮挡)
    mask = torch.rand(B, 1, num_patches_h, num_patches_w, device=img.device)
    mask = (mask < mask_ratio).float()
    
    # 上采样掩码到原图尺寸
    mask = F.interpolate(mask, size=(H, W), mode='nearest')
    
    # 应用掩码 (将被遮挡的区域置为 0)
    img_masked = img * (1 - mask)
    return img_masked

def test_model_robustness(model, model_name, img_size, image_path, patch_size=14):
    print(f"\n[{model_name}] 正在测试...")
    model.eval()
    
    # 1. 加载并处理真实图像
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        return
        
    try:
        # 定义图像预处理流程（缩放 -> 转为Tensor -> ImageNet标准归一化）
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 读取图像并增加 Batch 维度: [1, 3, H, W]
        img = Image.open(image_path).convert('RGB')
        img_orig = transform(img).unsqueeze(0) 
        
    except Exception as e:
        print(f"❌ 图片处理失败: {e}")
        return
        
    # 转移到 GPU
    if torch.cuda.is_available():
        img_orig = img_orig.cuda()
        model = model.cuda()
    
    # 2. 生成掩码图像 (遮挡比例 50%)
    img_masked = generate_masked_image(img_orig, mask_ratio=0.5, patch_size=patch_size)
    
    # 3. 进行推理
    with torch.no_grad():
        out_orig = model(img_orig)['cls_token'].squeeze(1)   # [B, Dim]
        out_masked = model(img_masked)['cls_token'].squeeze(1) # [B, Dim]

    # 4. 计算指标
    # --- 余弦相似度与余弦损失 ---
    cos_sim = F.cosine_similarity(out_orig, out_masked, dim=-1).mean().item()
    cos_loss = 1.0 - cos_sim
    
    # --- 特征方差 ---
    var_orig = torch.var(out_orig, dim=-1).mean().item()
    var_masked = torch.var(out_masked, dim=-1).mean().item()
    
    # --- 相对特征距离 (L2 Loss) ---
    l2_loss = F.mse_loss(out_orig, out_masked).item()

    print(f"  ├─ 图像尺寸: {img_size}x{img_size} (掩码率 50%, Patch Size: {patch_size})")
    print(f"  ├─ 余弦相似度: {cos_sim:.4f} (Cosine Loss: {cos_loss:.4f})")
    print(f"  ├─ 原始特征方差: {var_orig:.6f}")
    print(f"  ├─ 遮挡特征方差: {var_masked:.6f}")
    print(f"  └─ MSE (L2) 距离: {l2_loss:.6f}")


# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print(" 视觉大模型掩码鲁棒性测试 (CLS Token) - 真实图像")
    print("=" * 50)
    
    # 指定真实图片路径
    real_image_path = 'data/example.png'
    
    try:
        # EyeCLIP 默认使用 224x224，Patch=32
        eyeclip_model = EyeCLIP()
        test_model_robustness(eyeclip_model, "EyeCLIP (ViT-Base)", img_size=224, image_path=real_image_path, patch_size=32)
    except Exception as e:
        print(f"\n❌ EyeCLIP 实例化/测试失败: {e}")

    try:
        # RETFound 默认使用 518x518，Patch=14
        retfound_model = RETFound()
        test_model_robustness(retfound_model, "RETFound (ViT-Large)", img_size=518, image_path=real_image_path, patch_size=14)
    except Exception as e:
        print(f"\n❌ RETFound 实例化/测试失败: {e}")

    try:
        # DINOv2Giant 默认使用 518x518，Patch=14
        giant_model = DINOv2Giant()
        test_model_robustness(giant_model, "DINOv2 Giant (ViT-Giant)", img_size=518, image_path=real_image_path, patch_size=14)
    except Exception as e:
        print(f"\n❌ DINOv2 Giant 实例化/测试失败: {e}")