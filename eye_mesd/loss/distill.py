import torch.nn as nn
import torch.nn.functional as F
import math

class KLDiv(nn.Module):
    # 【修改点 1】增加 reduction 参数，默认保持原来的 batchmean
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super(KLDiv, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z_s, z_t, **kwargs):
        log_pred_student = F.log_softmax(z_s / self.temperature, dim=-1)
        pred_teacher = F.softmax(z_t / self.temperature, dim=-1)

        # 【修改点 2】根据 reduction 决定是否聚合
        if self.reduction == 'none':
            # 返回每张图独立的散度，形状为 [Batch]
            # kl_div 在 none 模式下返回 [Batch, Dim]，我们需要把特征维度的差异加起来
            kl = F.kl_div(log_pred_student, pred_teacher, reduction='none')
            kd_loss = kl.sum(dim=-1) * (self.temperature ** 2)
        else:
            # 正常返回标量，用于反向传播更新 Backbone
            kd_loss = F.kl_div(log_pred_student, pred_teacher, reduction=self.reduction) * (self.temperature ** 2)
        
        return kd_loss


class Cosine(nn.Module):
    # 【修改点 3】增加 reduction 参数
    def __init__(self, dim=-1, reduction='mean', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cos = nn.CosineSimilarity(dim=dim)
        self.reduction = reduction
    
    def forward(self, student_out, teacher_out):
        # cos 算出来的形状是 [Batch] (对于 CLS token)
        loss = 1 - self.cos(student_out, teacher_out)
        
        # 【修改点 4】根据 reduction 决定是否平均
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            # 如果输入是三维的 Patch Token [B, N, C]，算出来是 [B, N]
            # 对于 RL 我们需要单张图的损失，所以沿着 N (Patch数) 求平均，变成 [B]
            if loss.dim() > 1:
                return loss.mean(dim=1)
            return loss # 如果是 CLS Token，直接返回 [B]
        return loss


class FeatureLoss(nn.Module):
    # 【修改点 5】支持传递 reduction 给内部函数
    def __init__(self, alpha=0.9, beta=0.1, dim=-1, reduction='mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.cos_fn = Cosine(dim=dim, reduction=reduction)
        # SmoothL1 也要听指挥
        self.smoothl1 = nn.SmoothL1Loss(reduction=reduction)
    
    def forward(self, student_out, teacher_out):
        """
        student_out: (B, N1, C)
        teacher_out: (B, N2, C)
        """
        N1 = student_out.shape[1]
        B, N2, C = teacher_out.shape
        l1 = int(math.sqrt(N1))
        l2 = int(math.sqrt(N2))
        teacher_out = teacher_out.permute(0, 2, 1).view(B, C, l2, l2)
        teacher_out = F.interpolate(teacher_out, size=(l1, l1), mode='bicubic')
        teacher_out = teacher_out.view(B, C, -1).permute(0, 2, 1)
        
        cos = self.cos_fn(student_out, teacher_out) * self.alpha
        
        if self.reduction == 'none':
            # smoothl1 算出 [B, N1, C]，需要转成单张图的 loss [B]
            l1_loss = self.smoothl1(student_out, teacher_out).mean(dim=(1, 2))
        else:
            l1_loss = self.smoothl1(student_out, teacher_out)
            
        smoothl1 = l1_loss * self.beta
        return cos + smoothl1


# 【修改点 6】允许实例化时传入 reduction 状态
def get_kd_loss(name, reduction='mean'):
    if name.lower() == 'kldiv':
        loss = KLDiv(temperature=1.0, reduction=reduction)
    elif name.lower() == 'radio':
        loss = Cosine(reduction=reduction)
    else:
        raise NotImplementedError
    return loss