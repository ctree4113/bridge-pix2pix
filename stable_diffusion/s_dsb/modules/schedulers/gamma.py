import torch
import numpy as np
from typing import Union

class GammaScheduler:
    """S-DSB的gamma调度器"""
    def __init__(
        self,
        schedule: str = "linear",
        num_timesteps: int = 1000,
        gamma_min: float = 1e-4,
        gamma_max: float = 0.1
    ):
        self.schedule = schedule
        self.num_timesteps = num_timesteps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        self.prepare_schedule()
        
    def prepare_schedule(self):
        """准备gamma schedule"""
        if self.schedule == "linear":
            # 线性调度
            gammas = torch.linspace(self.gamma_min, self.gamma_max, 
                                  self.num_timesteps // 2)
            self.gammas = torch.cat([gammas, gammas.flip(dims=(0,))])
        elif self.schedule == "cosine":
            # 余弦调度
            steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
            alpha_bar = torch.cos(((steps / self.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            self.gammas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 
                                   self.gamma_min, self.gamma_max)
        else:
            raise ValueError(f"Unknown schedule type {self.schedule}")
            
    def __call__(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """获取指定时间步的gamma值"""
        return self.gammas[t]