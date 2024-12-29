import torch
import abc
from typing import Optional, Union, List, Tuple, Dict, Any

class BaseSampler(abc.ABC):
    """采样器基类"""
    
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
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
        pass
        
    @abc.abstractmethod
    def p_sample(
        self,
        model: Any,
        x: torch.Tensor,
        t: Union[torch.Tensor, int],
        cond: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """单步采样
        
        Args:
            model: S-DSB模型
            x: 当前状态
            t: 时间步
            cond: 条件信息
            
        Returns:
            下一个状态
        """
        pass