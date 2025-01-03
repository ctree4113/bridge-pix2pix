import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from ...modules.schedulers.gamma import AdaptiveGammaScheduler
from ...util import extract

class SDSBCore:
    """S-DSB核心功能实现"""
    
    def __init__(
        self,
        num_timesteps: int,
        gamma_min: float,
        gamma_max: float,
        gamma_schedule: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        # 初始化gamma调度器
        self.gamma_scheduler = AdaptiveGammaScheduler(
            num_timesteps=num_timesteps,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            schedule=gamma_schedule
        )
        
    def compute_forward_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """计算正向漂移项 f_k(x_k)"""
        # 获取gamma值
        gamma_t = extract(self.gamma_scheduler.gammas, t, x.shape)
        
        # 计算漂移项
        drift = -0.5 * gamma_t * x
        return drift
        
    def compute_backward_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """计算反向漂移项 b_k(x_k)"""
        # 获取gamma值
        gamma_t = extract(self.gamma_scheduler.gammas, t, x.shape)
        
        # 计算漂移项
        drift = 0.5 * gamma_t * x
        return drift
        
    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """正向扩散过程
        
        实现文档中的正向扩散公式:
        x_{k+1} = x_k + γ_k f_k(x_k) + √(2γ_k) ε_k
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        gamma_t = extract(self.gamma_scheduler.gammas, t, x_0.shape)
        
        # 计算漂移项
        drift = self.compute_forward_drift(x_0, t)
        
        # 正向扩散
        x_t = x_0 + gamma_t * drift + torch.sqrt(2 * gamma_t) * noise
        
        return x_t, noise
        
    def backward_process(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """反向去噪过程
        
        实现文档中的反向扩散公式:
        x_k = x_{k+1} + γ_k b_k(x_{k+1}) + √(2γ_k) ε_k
        """
        gamma_t = extract(self.gamma_scheduler.gammas, t, x_t.shape)
        
        # 计算反向漂移项
        drift = self.compute_backward_drift(x_t, t, cond)
        
        # 生成噪声
        noise = torch.randn_like(x_t)
        
        # 反向去噪
        x_prev = x_t + gamma_t * drift + torch.sqrt(2 * gamma_t) * noise
        
        return x_prev
        
    def compute_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算简化的损失函数
        
        实现文档中的简化损失:
        L_{F_k} = E_{(x_k, x_{k+1})} ||F_k(x_k) - x_{k+1}||^2
        L_{B_k} = E_{(x_k, x_{k+1})} ||B_k(x_{k+1}) - x_k||^2
        """
        # 正向过程
        x_t, noise = self.forward_process(x_0, t, noise)
        
        # 反向预测
        noise_pred = self.predict_noise(x_t, t, cond)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        return loss 