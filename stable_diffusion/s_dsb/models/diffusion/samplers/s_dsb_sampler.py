import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict, Any
from tqdm import tqdm

from .base import BaseSampler
from .utils import extract, noise_like
from .s_dsb_core import SDSBCore

class SDSBSampler(BaseSampler):
    """改进的S-DSB采样器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.s_dsb = SDSBCore(**kwargs)
        
    def p_sample(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """单步采样，支持双向过程"""
        # 获取方向
        direction = kwargs.get("direction", "backward")
        
        if direction == "forward":
            return self.s_dsb.forward_process(x, t)[0]
        else:
            return self.s_dsb.backward_process(x, t, cond)
        
    def sample(
        self,
        model: Any,
        shape: Tuple[int, ...],
        cond: Optional[Dict[str, torch.Tensor]] = None,
        x_start: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """采样过程
        
        Args:
            model: S-DSB模型
            shape: 输出形状
            cond: 条件信息
            x_start: 起始图像
            noise: 初始噪声
            
        Returns:
            采样结果和可选的中间状态
        """
        device = next(model.parameters()).device
        
        # 初始化
        if noise is None:
            noise = torch.randn(shape, device=device)
        x = noise
        
        # 存储中间结果
        intermediates = []
        
        # 逐步采样
        with torch.no_grad():
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling'):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x = self.p_sample(
                    model,
                    x,
                    t,
                    cond=cond,
                    clip_denoised=self.clip_denoised,
                    **kwargs
                )
                
                if self.return_intermediates:
                    intermediates.append(x.clone())
                    
        if self.return_intermediates:
            return x, intermediates
        return x
