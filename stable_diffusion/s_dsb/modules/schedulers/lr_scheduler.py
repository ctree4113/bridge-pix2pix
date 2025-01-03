import numpy as np

class LRScheduler:
    """学习率调度器"""
    
    def __init__(
        self,
        warm_up_steps: int,
        total_steps: int,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        verbose: bool = False
    ):
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.verbose = verbose
        
    def __call__(self, step: int) -> float:
        """获取当前步骤的学习率"""
        if step < self.warm_up_steps:
            # 预热阶段：线性增加
            lr = self.base_lr * (step / self.warm_up_steps)
        else:
            # 余弦退火
            progress = (step - self.warm_up_steps) / (self.total_steps - self.warm_up_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
            
        if self.verbose:
            print(f"Step {step}: lr = {lr}")
            
        return lr