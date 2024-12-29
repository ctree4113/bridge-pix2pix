import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FlowReparameterizer(nn.Module):
    """Flow重参数化实现
    
    Flow重参数化通过将输入映射到标准正态分布来实现分布转换。
    前向过程: x_t -> (x_t - x_0) / sqrt(gamma_t)
    反向过程: x_t -> (x_t - x_1) / sqrt(1 - gamma_t)
    """
    
    def __init__(
        self,
        normalize: bool = True,
        scale_factor: float = 1.0,
        eps: float = 1e-8,
        clip_value: Optional[float] = None
    ):
        """初始化
        
        Args:
            normalize: 是否对输入进行归一化
            scale_factor: 缩放因子
            eps: 数值稳定性的小值
            clip_value: 裁剪值范围,None表示不裁剪
        """
        super().__init__()
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.eps = eps
        self.clip_value = clip_value
        
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """输入归一化
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            归一化后的张量
        """
        if not self.normalize:
            return x
            
        # 计算每个样本的均值和标准差
        b, c, h, w = x.shape
        x_flat = x.view(b, -1)
        mean = x_flat.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
        std = x_flat.std(dim=1, keepdim=True).view(b, 1, 1, 1)
        
        # 归一化
        x = (x - mean) / (std + self.eps)
        return x
        
    def forward_reparameterize(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        gamma_t: torch.Tensor
    ) -> torch.Tensor:
        """前向重参数化
        
        Args:
            x_t: 当前时间步的输入 [B, C, H, W]
            x_0: 起始图像 [B, C, H, W]
            gamma_t: gamma值 [B]
            
        Returns:
            重参数化后的张量
        """
        # 展开gamma维度
        gamma_t = gamma_t.view(-1, 1, 1, 1)
        
        # 归一化输入
        x_t = self.normalize_input(x_t)
        x_0 = self.normalize_input(x_0)
        
        # 计算重参数化
        z = (x_t - x_0) / (torch.sqrt(gamma_t) + self.eps)
        
        # 缩放
        z = z * self.scale_factor
        
        # 可选的裁剪
        if self.clip_value is not None:
            z = torch.clamp(z, -self.clip_value, self.clip_value)
            
        return z
        
    def backward_reparameterize(
        self,
        x_t: torch.Tensor,
        x_1: torch.Tensor,
        gamma_t: torch.Tensor
    ) -> torch.Tensor:
        """反向重参数化
        
        Args:
            x_t: 当前时间步的输入 [B, C, H, W]
            x_1: 目标图像 [B, C, H, W]
            gamma_t: gamma值 [B]
            
        Returns:
            重参数化后的张量
        """
        # 展开gamma维度
        gamma_t = gamma_t.view(-1, 1, 1, 1)
        
        # 归一化输入
        x_t = self.normalize_input(x_t)
        x_1 = self.normalize_input(x_1)
        
        # 计算重参数化
        z = (x_t - x_1) / (torch.sqrt(1 - gamma_t) + self.eps)
        
        # 缩放
        z = z * self.scale_factor
        
        # 可选的裁剪
        if self.clip_value is not None:
            z = torch.clamp(z, -self.clip_value, self.clip_value)
            
        return z
        
    def inverse_forward(
        self,
        z: torch.Tensor,
        x_0: torch.Tensor,
        gamma_t: torch.Tensor
    ) -> torch.Tensor:
        """前向重参数化的逆变换
        
        Args:
            z: 重参数化后的张量 [B, C, H, W]
            x_0: 起始图像 [B, C, H, W]
            gamma_t: gamma值 [B]
            
        Returns:
            原始空间的张量
        """
        # 展开gamma维度
        gamma_t = gamma_t.view(-1, 1, 1, 1)
        
        # 归一化输入
        x_0 = self.normalize_input(x_0)
        
        # 反缩放
        z = z / self.scale_factor
        
        # 逆变换
        x_t = z * torch.sqrt(gamma_t) + x_0
        
        return x_t
        
    def inverse_backward(
        self,
        z: torch.Tensor,
        x_1: torch.Tensor,
        gamma_t: torch.Tensor
    ) -> torch.Tensor:
        """反向重参数化的逆变换
        
        Args:
            z: 重参数化后的张量 [B, C, H, W]
            x_1: 目标图像 [B, C, H, W]
            gamma_t: gamma值 [B]
            
        Returns:
            原始空间的张量
        """
        # 展开gamma维度
        gamma_t = gamma_t.view(-1, 1, 1, 1)
        
        # 归一化输入
        x_1 = self.normalize_input(x_1)
        
        # 反缩放
        z = z / self.scale_factor
        
        # 逆变换
        x_t = z * torch.sqrt(1 - gamma_t) + x_1
        
        return x_t
        
    def forward(
        self,
        x_t: torch.Tensor,
        x_ref: torch.Tensor,
        gamma_t: torch.Tensor,
        direction: str = "forward"
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x_t: 当前时间步的输入 [B, C, H, W]
            x_ref: 参考图像(x_0或x_1) [B, C, H, W]
            gamma_t: gamma值 [B]
            direction: 方向("forward" or "backward")
            
        Returns:
            重参数化后的张量
        """
        if direction == "forward":
            return self.forward_reparameterize(x_t, x_ref, gamma_t)
        else:
            return self.backward_reparameterize(x_t, x_ref, gamma_t)