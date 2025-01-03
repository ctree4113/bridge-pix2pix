from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SDSBConfig:
    """S-DSB配置"""
    num_timesteps: int = 1000
    gamma_min: float = 1e-4
    gamma_max: float = 0.1
    gamma_schedule: str = "linear"
    reparam_type: str = "flow"
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # 模型配置
    model_channels: int = 96
    attention_resolutions: List[int] = (4,)
    channel_mult: List[int] = (1, 2, 3)
    num_heads: int = 4
    context_dim: int = 256 