import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class SDSBLoss(nn.Module):
    """S-DSB损失函数"""
    
    def __init__(self, num_projections: int = 10):
        super().__init__()
        self.num_projections = num_projections
        
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        noise: Optional[torch.Tensor] = None,
        direction: str = "forward"
    ) -> torch.Tensor:
        """计算损失"""
        # 获取gamma值
        gamma_t = model.core.gamma_scheduler(t)
        
        # 正向过程
        if direction == "forward":
            x_t, eps = model.forward_process(x, t, noise)
            pred = model.predict_noise(x_t, t, cond)
            loss = F.mse_loss(pred, eps)
            
        # 反向过程
        else:
            x_t = model.backward_process(x, t, cond)
            pred = model.predict_start(x_t, t, cond)
            loss = F.mse_loss(pred, x)
            
        return loss