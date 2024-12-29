import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class SDSBLoss(nn.Module):
    """S-DSB的损失函数实现"""
    def __init__(
        self,
        reparam_type: str = "flow",
        lambda_simple: float = 1.0,
        lambda_vlb: float = 0.001,
        lambda_perceptual: Optional[float] = None,
        lambda_clip: Optional[float] = None,
        perceptual_model = None,
        clip_model = None,
        use_huber: bool = False,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.reparam_type = reparam_type
        self.lambda_simple = lambda_simple
        self.lambda_vlb = lambda_vlb
        self.lambda_perceptual = lambda_perceptual
        self.lambda_clip = lambda_clip
        
        # 可选的感知损失
        self.perceptual_model = perceptual_model
        self.clip_model = clip_model
        
        # 损失函数配置
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        
    def compute_simple_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算简单重建损失"""
        if self.use_huber:
            return F.huber_loss(pred, target, delta=self.huber_delta)
        return F.mse_loss(pred, target)
        
    def compute_vlb_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma_t: torch.Tensor,
        direction: str = "forward"
    ) -> torch.Tensor:
        """计算变分下界损失"""
        if direction == "forward":
            log_gamma = torch.log(gamma_t)
            weight = gamma_t
        else:
            log_gamma = torch.log(1 - gamma_t)
            weight = 1 - gamma_t
            
        return 0.5 * (-log_gamma + weight * torch.pow(pred - target, 2).mean())
        
    def compute_perceptual_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算感知损失"""
        if self.perceptual_model is None:
            return torch.tensor(0.0, device=pred.device)
            
        return self.perceptual_model(pred, target)
        
    def compute_clip_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算CLIP损失"""
        if self.clip_model is None:
            return torch.tensor(0.0, device=pred.device)
            
        # 获取CLIP特征
        pred_features = self.clip_model.encode_image(pred)
        target_features = self.clip_model.encode_image(target)
        
        # 计算余弦相似度损失
        pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)
        
        return 1 - (pred_features * target_features).sum(dim=-1).mean()
        
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        gamma_t: torch.Tensor,
        direction: str = "forward",
        x_t: Optional[torch.Tensor] = None,
        x_start: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            model_output: 模型输出
            target: 目标值
            gamma_t: gamma值
            direction: 方向("forward" or "backward")
            x_t: 当前时间步的输入
            x_start: 起始/终止图像
            
        Returns:
            loss: 总损失
            log_dict: 日志字典
        """
        prefix = "forward" if direction == "forward" else "backward"
        
        # 计算各项损失
        simple_loss = self.compute_simple_loss(model_output, target)
        vlb_loss = self.compute_vlb_loss(model_output, target, gamma_t, direction)
        perceptual_loss = self.compute_perceptual_loss(model_output, target)
        clip_loss = self.compute_clip_loss(model_output, target)
        
        # 总损失
        loss = self.lambda_simple * simple_loss + self.lambda_vlb * vlb_loss
        
        if self.lambda_perceptual is not None:
            loss = loss + self.lambda_perceptual * perceptual_loss
            
        if self.lambda_clip is not None:
            loss = loss + self.lambda_clip * clip_loss
            
        # 日志
        log_dict = {
            f"train/{prefix}_simple_loss": simple_loss.detach().mean(),
            f"train/{prefix}_vlb_loss": vlb_loss.detach().mean(),
        }
        
        if self.lambda_perceptual is not None:
            log_dict[f"train/{prefix}_perceptual_loss"] = perceptual_loss.detach().mean()
            
        if self.lambda_clip is not None:
            log_dict[f"train/{prefix}_clip_loss"] = clip_loss.detach().mean()
            
        return loss, log_dict