import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional, Union, List, Tuple, Any

# 导入核心模块
from .s_dsb_core import SDSBCore
from .s_dsb_diffusion import SDSBDiffusion
from ...modules.diffusionmodules.s_dsb_unet import SDSBUNet
from ...modules.encoders.modules import FrozenCLIPEmbedder
from ...modules.ema import LitEma
from ...modules.losses.s_dsb_losses import SDSBLoss
from ...util import exists, default, instantiate_from_config, extract
from ...modules.reparam.flow import reparameterize_flow
from ...schedulers.lr_scheduler import LambdaLinearScheduler

class SDSB(pl.LightningModule):
    """基础S-DSB实现"""
    
    def __init__(
        self,
        unet_config: Dict,
        clip_config: Optional[Dict] = None,
        num_timesteps: int = 1000,
        gamma_min: float = 1e-4,
        gamma_max: float = 0.1,
        gamma_schedule: str = "linear",
        learning_rate: float = 1e-4,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        loss_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化U-Net和CLIP
        self.unet = instantiate_from_config(unet_config)
        if clip_config:
            clip_params = clip_config.get("params", {})
            self.clip = FrozenCLIPEmbedder(**clip_params)
        else:
            self.clip = None
        
        # 初始化核心扩散模块
        self.core = SDSBCore(
            num_timesteps=num_timesteps,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_schedule=gamma_schedule
        )
        
        # 初始化扩散过程
        self.diffusion = SDSBDiffusion(
            unet=self.unet,
            core=self.core,
            clip=self.clip
        )
        
        # 初始化损失函数
        if loss_config:
            self.loss_fn = instantiate_from_config(loss_config)
        else:
            self.loss_fn = SDSBLoss()
            
        # EMA
        if use_ema:
            self.model_ema = LitEma(self.diffusion, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            
        self.use_ema = use_ema
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[Dict] = None) -> torch.Tensor:
        """前向传播"""
        return self.diffusion.forward_process(x, t, cond)[0]
        
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        cond: Optional[Dict] = None,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """采样生成"""
        shape = (batch_size, self.unet.in_channels, *self.image_size)
        
        # 使用EMA模型采样
        if self.use_ema:
            with self.ema_scope():
                return self.diffusion.sample(shape, cond, return_intermediates=return_intermediates)
                
        return self.diffusion.sample(shape, cond, return_intermediates=return_intermediates)
        
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        x = batch["image"]
        
        # 随机时间步
        t = torch.randint(0, self.core.num_timesteps, (x.shape[0],), device=self.device)
        
        # 获取条件
        cond = self.get_input(batch, "condition") if "condition" in batch else None
        
        # 计算损失
        loss = self.loss_fn(self.diffusion, x, t, cond)
        
        # 记录日志
        self.log("train/loss", loss, prog_bar=True, logger=True)
        
        return loss
        
    def on_train_batch_end(self, *args, **kwargs):
        """训练批次结束后更新EMA"""
        if self.use_ema:
            self.model_ema(self.diffusion)
            
    @contextmanager
    def ema_scope(self, context=None):
        """EMA上下文管理器"""
        if self.use_ema:
            self.model_ema.store(self.diffusion.parameters())
            self.model_ema.copy_to(self.diffusion)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.diffusion.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
                    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        params = list(self.diffusion.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        
        scheduler = LambdaLinearScheduler(
            optimizer=opt,
            warm_up_steps=self.hparams.warmup_steps,
            total_steps=self.hparams.max_steps
        )
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }