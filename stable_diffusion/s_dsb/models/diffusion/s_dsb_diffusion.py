import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List, Any

class SDSBDiffusion(nn.Module):
    """S-DSB扩散过程的实现"""
    
    def __init__(
        self,
        unet: nn.Module,
        core: nn.Module,
        clip: Optional[nn.Module] = None
    ):
        super().__init__()
        self.unet = unet
        self.core = core
        self.clip = clip
        
    def forward_process(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """正向扩散过程"""
        return self.core.forward_process(x, t, noise)
        
    def backward_process(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """反向去噪过程"""
        return self.core.backward_process(x, t, cond)
        
    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """预测噪声"""
        # 获取CLIP条件嵌入
        clip_embed = self.clip.encode(cond["text"]) if cond else None
        
        # 预测噪声
        noise_pred = self.unet(x, t, clip_embed)
        return noise_pred
        
    def predict_start(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """预测起始图像"""
        # 获取gamma值
        gamma_t = self.core.gamma_scheduler(t)
        
        # 预测噪声
        noise_pred = self.predict_noise(x, t, cond)
        
        # 计算起始图像
        x_0 = x - torch.sqrt(gamma_t) * noise_pred
        return x_0
        
    def compute_forward_drift(
        self, 
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: float = 5.0
    ) -> torch.Tensor:
        """计算正向漂移项 f_k(x_k)"""
        # 获取CLIP条件嵌入
        clip_embed = self.clip.encode(cond["text"]) if cond else None
        
        # 预测噪声
        noise_pred = self.unet(x, t, clip_embed)
        
        # 计算漂移项
        gamma_t = extract(self.core.gamma_scheduler.gammas, t, x.shape)
        drift = -0.5 * gamma_t * noise_pred
        
        return drift
        
    def compute_backward_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: float = 5.0
    ) -> torch.Tensor:
        """计算反向漂移项 b_k(x_k)"""
        # 获取CLIP条件嵌入
        clip_embed = self.clip.encode(cond["text"]) if cond else None
        
        # 预测噪声
        noise_pred = self.unet(x, t, clip_embed)
        
        # 应用classifier-free guidance
        if cond:
            noise_pred_uncond = self.unet(x, t, None)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
        # 计算漂移项
        gamma_t = extract(self.core.gamma_scheduler.gammas, t, x.shape)
        drift = 0.5 * gamma_t * noise_pred
        
        return drift 