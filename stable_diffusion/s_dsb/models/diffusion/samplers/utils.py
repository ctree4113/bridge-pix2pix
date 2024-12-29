import torch
from typing import Tuple, Union

def extract(
    a: torch.Tensor,
    t: Union[torch.Tensor, int],
    x_shape: Tuple[int, ...]
) -> torch.Tensor:
    """从batch中提取指定时间步的值
    
    Args:
        a: 源张量
        t: 时间步
        x_shape: 目标形状
        
    Returns:
        提取的值
    """
    batch_size = t.shape[0] if isinstance(t, torch.Tensor) else 1
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def noise_like(
    shape: Tuple[int, ...],
    device: torch.device,
    repeat: bool = False
) -> torch.Tensor:
    """生成噪声张量
    
    Args:
        shape: 形状
        device: 设备
        repeat: 是否重复第一维
        
    Returns:
        噪声张量
    """
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    else:
        return torch.randn(shape, device=device)
