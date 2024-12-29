import numpy as np
from typing import List

class LambdaWarmUpCosineScheduler:
    """余弦预热学习率调度器"""
    def __init__(
        self,
        warm_up_steps: int,
        lr_min: float,
        lr_max: float,
        lr_start: float,
        max_decay_steps: int,
        verbosity_interval: int = 0
    ):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def __call__(self, n: int, **kwargs) -> float:
        """计算学习率"""
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr: {self.last_lr}")
                
        # 预热阶段
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
            
        # 衰减阶段
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

class LambdaLinearScheduler:
    """线性学习率调度器"""
    def __init__(
        self,
        warm_up_steps: List[int],
        f_min: List[float],
        f_max: List[float],
        f_start: List[float],
        cycle_lengths: List[int],
        verbosity_interval: int = 0
    ):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n: int) -> int:
        """找到当前所在的周期"""
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1
        return interval

    def __call__(self, n: int, **kwargs) -> float:
        """计算学习率"""
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"step: {n}, lr: {self.last_f}, cycle: {cycle}")

        # 预热阶段
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
            
        # 线性衰减阶段
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n) / self.cycle_lengths[cycle]
            self.last_f = f
            return f