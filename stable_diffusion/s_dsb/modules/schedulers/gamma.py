import torch
import numpy as np
from typing import Union

class AdaptiveGammaScheduler:
    """自适应gamma调度器"""
    
    def __init__(
        self,
        num_timesteps: int,
        gamma_min: float,
        gamma_max: float,
        schedule: str = "linear",
        adaptive: bool = True
    ):
        self.num_timesteps = num_timesteps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.schedule = schedule
        self.adaptive = adaptive
        
        self.prepare_schedule()
        
    def prepare_schedule(self):
        """准备gamma schedule"""
        if self.schedule == "linear":
            gammas = torch.linspace(
                self.gamma_min,
                self.gamma_max,
                self.num_timesteps // 2
            )
            self.gammas = torch.cat([gammas, gammas.flip(dims=(0,))])
            
        elif self.schedule == "cosine":
            steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
            alpha_bar = torch.cos(
                ((steps / self.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5
            ) ** 2
            self.gammas = torch.clip(
                1 - alpha_bar[1:] / alpha_bar[:-1],
                self.gamma_min,
                self.gamma_max
            )
            
    def __call__(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """获取gamma值"""
        # 确保 gammas 在正确的设备上
        if isinstance(t, torch.Tensor):
            self.gammas = self.gammas.to(t.device)
        gamma = self.gammas[t]
        
        if self.adaptive:
            # 根据训练进度调整gamma
            progress = t.float() / self.num_timesteps
            gamma = gamma * (1 - 0.5 * progress)
            
        return gamma