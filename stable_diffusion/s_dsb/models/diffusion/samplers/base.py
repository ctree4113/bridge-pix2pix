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
        """采样过程"""
        pass
        
    @abc.abstractmethod
    def p_sample(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """单步采样"""
        pass
        
    def sample_progressive(
        self,
        model: Any,
        shape: Tuple[int, ...],
        steps: List[int],
        **kwargs
    ) -> List[torch.Tensor]:
        """渐进式采样"""
        samples = []
        x = None
        
        for step in steps:
            x = self.sample(model, shape, x_start=x, **kwargs)
            samples.append(x)
            
        return samples