import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict, Any
from tqdm import tqdm

from .base import BaseSampler
from .utils import extract, noise_like

class SDSBSampler(BaseSampler):
    """S-DSB采样器实现"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        gamma_scheduler = None,
        reparam_type: str = "flow",
        use_ema: bool = True,
        clip_denoised: bool = True,
        return_intermediates: bool = False
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.gamma_scheduler = gamma_scheduler
        self.reparam_type = reparam_type
        self.use_ema = use_ema
        self.clip_denoised = clip_denoised
        self.return_intermediates = return_intermediates
        
    def p_sample(
        self,
        model: Any,
        x: torch.Tensor,
        t: Union[torch.Tensor, int],
        cond: Optional[Dict[str, torch.Tensor]] = None,
        clip_denoised: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """单步采样
        
        Args:
            model: S-DSB模型
            x: 当前状态
            t: 时间步
            cond: 条件信息
            clip_denoised: 是否裁剪去噪结果
            
        Returns:
            下一个状态
        """
        # 获取gamma值
        gamma_t = extract(self.gamma_scheduler.gammas, t, x.shape)
        
        # 模型预测
        with torch.no_grad():
            if self.use_ema:
                with model.ema_scope():
                    pred = model(x, t, cond)
            else:
                pred = model(x, t, cond)
                
        # Flow重参数化
        if self.reparam_type == "flow":
            # 添加预测的噪声
            x = x + torch.sqrt(gamma_t) * pred
            
            # 可选裁剪
            if clip_denoised:
                x = x.clamp(-1., 1.)
                
        # Terminal重参数化
        else:
            x = pred
            if clip_denoised:
                x = x.clamp(-1., 1.)
                
        # 添加噪声(如果不是最后一步)
        if t[0] > 0:
            noise = torch.randn_like(x)
            next_gamma = extract(self.gamma_scheduler.gammas, t-1, x.shape)
            x = x + torch.sqrt(next_gamma) * noise
            
        return x
        
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
