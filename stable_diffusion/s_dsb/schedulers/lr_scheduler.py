import torch
from torch.optim.lr_scheduler import _LRScheduler

class LambdaLinearScheduler(_LRScheduler):
    def __init__(self, optimizer, warm_up_steps, total_steps, last_epoch=-1):
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warm_up_steps:
            # 预热阶段: 线性增加
            alpha = self.last_epoch / self.warm_up_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 预热后: 线性衰减
            alpha = (self.total_steps - self.last_epoch) / (self.total_steps - self.warm_up_steps)
            alpha = max(0.0, alpha)
            return [base_lr * alpha for base_lr in self.base_lrs] 