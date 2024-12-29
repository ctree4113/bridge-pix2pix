import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional, Union, List, Tuple

from s_dsb.modules.diffusionmodules.s_dsb_unet import SDSBUNet
from s_dsb.modules.encoders.modules import FrozenCLIPEmbedder
from s_dsb.modules.ema import LitEma
from s_dsb.util import exists, default, instantiate_from_config
from ...modules.schedulers.gamma import GammaScheduler

class SDSB(pl.LightningModule):
    """Simplified Diffusion Schrödinger Bridge (S-DSB)的实现"""
    
    def __init__(
        self,
        unet_config,
        clip_config=None,
        num_timesteps=1000,
        gamma_min=1e-4,
        gamma_max=0.02,
        gamma_schedule="linear",
        reparam_type="flow",
        use_ema=True,
        ema_decay=0.9999,
        learning_rate=1e-4,
        **kwargs
    ):
        super().__init__()
        
        # 基础配置
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.use_ema = use_ema
        self.reparam_type = reparam_type
        
        # 初始化UNet
        # 过滤UNet配置参数
        unet_params = dict(unet_config.get("params", {}))
        # 移除不支持的参数
        supported_params = [
            "image_size", "in_channels", "model_channels", 
            "out_channels", "num_res_blocks", "attention_resolutions",
            "dropout", "channel_mult", "conv_resample", 
            "dims", "num_heads", "num_head_channels", "num_heads_upsample",
            "use_scale_shift_norm", "resblock_updown", "use_new_attention_order"
        ]
        filtered_params = {k: v for k, v in unet_params.items() if k in supported_params}
        
        # 为forward模型创建配置
        forward_config = dict(unet_config)
        forward_config["params"] = dict(filtered_params)
        forward_config["params"].update({
            "reparam_type": reparam_type,
            "direction": "forward"
        })
        self.forward_model = instantiate_from_config(forward_config)
        
        # 为backward模型创建配置
        backward_config = dict(unet_config)
        backward_config["params"] = dict(filtered_params)
        backward_config["params"].update({
            "reparam_type": reparam_type,
            "direction": "backward"
        })
        self.backward_model = instantiate_from_config(backward_config)
        
        # 初始化CLIP
        if exists(clip_config):
            self.cond_stage_model = instantiate_from_config(clip_config)
            print("Initialized CLIP encoder for conditional generation")
        else:
            self.cond_stage_model = None
            print("No CLIP config provided, running in unconditional mode")
            
        # EMA支持
        if self.use_ema:
            self.forward_model_ema = LitEma(self.forward_model, decay=ema_decay)
            self.backward_model_ema = LitEma(self.backward_model, decay=ema_decay)
            print(f"Using EMA with decay={ema_decay}")
            
        # gamma调度器
        self.gamma_scheduler = GammaScheduler(
            schedule=gamma_schedule,
            num_timesteps=num_timesteps,
            gamma_min=gamma_min,
            gamma_max=gamma_max
        )
        
        # 注册buffer
        self.register_buffer('gammas', self.gamma_scheduler.gammas)
        
    def get_input(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """处理输入数据"""
        x_0 = batch["source"]
        x_1 = batch["target"]
        
        # 处理条件信息
        c = None
        if exists(self.cond_stage_model) and exists(batch.get("caption")):
            c = self.get_learned_conditioning(batch["caption"])
            
        return x_0, x_1, c
        
    def get_learned_conditioning(self, c: str) -> Dict[str, torch.Tensor]:
        """获取条件编码"""
        if self.cond_stage_model is None:
            return None
        return {"c_crossattn": self.cond_stage_model(c)}
        
    @contextmanager
    def ema_scope(self, context: Optional[str] = None):
        """EMA上下文管理器"""
        if self.use_ema:
            self.forward_model_ema.store(self.forward_model.parameters())
            self.backward_model_ema.store(self.backward_model.parameters())
            self.forward_model_ema.copy_to(self.forward_model)
            self.backward_model_ema.copy_to(self.backward_model)
            if context is not None:
                print(f"{context}: Using EMA parameters")
        try:
            yield None
        finally:
            if self.use_ema:
                self.forward_model_ema.restore(self.forward_model.parameters())
                self.backward_model_ema.restore(self.backward_model.parameters())
                if context is not None:
                    print(f"{context}: Restored training parameters")
                    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """采样过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        gamma_t = self.gammas[t]
        gamma_t = gamma_t.view(-1, 1, 1, 1)
        return x_start + torch.sqrt(gamma_t) * noise
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict] = None,
        direction: str = "forward"
    ) -> torch.Tensor:
        """前向传播"""
        model = self.forward_model if direction == "forward" else self.backward_model
        return model(x, t, cond)
        
    def training_step(self, batch: Dict, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """训练步骤"""
        x_0, x_1, cond = self.get_input(batch)
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=self.device)
        
        if optimizer_idx == 0:  # 前向模型
            loss = self.compute_forward_loss(x_0, x_1, t, cond)
            self.log("train/forward_loss", loss, prog_bar=True)
        else:  # 反向模型
            loss = self.compute_backward_loss(x_0, x_1, t, cond)
            self.log("train/backward_loss", loss, prog_bar=True)
            
        # 添加CLIP损失
        if exists(self.cond_stage_model):
            text_embeddings = self.get_text_embeddings(batch)
            clip_loss = self.compute_clip_loss(x_0, x_1, text_embeddings)
            loss = loss + self.clip_weight * clip_loss
            self.log("train/clip_loss", clip_loss)
            
        return loss
        
    def compute_forward_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict] = None
    ) -> torch.Tensor:
        """计算前向损失"""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        if self.reparam_type == "flow":
            pred = self.forward_model(x_t, t, cond)
            gamma_t = self.gammas[t].view(-1, 1, 1, 1)
            target = (x_1 - x_t) / torch.sqrt(gamma_t)
        else:  # term
            pred = self.forward_model(torch.cat([x_t, x_0], dim=1), t, cond)
            target = x_1
            
        return F.mse_loss(pred, target)
        
    def compute_backward_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict] = None
    ) -> torch.Tensor:
        """计算反向损失"""
        noise = torch.randn_like(x_1)
        x_t = self.q_sample(x_1, t, noise)
        
        if self.reparam_type == "flow":
            pred = self.backward_model(x_t, t, cond)
            gamma_t = self.gammas[t].view(-1, 1, 1, 1)
            target = (x_0 - x_t) / torch.sqrt(gamma_t)
        else:  # term
            pred = self.backward_model(torch.cat([x_t, x_1], dim=1), t, cond)
            target = x_0
            
        return F.mse_loss(pred, target)
        
    def configure_optimizers(self):
        """配置优化器"""
        opt_forward = torch.optim.AdamW(
            self.forward_model.parameters(),
            lr=self.learning_rate
        )
        opt_backward = torch.optim.AdamW(
            self.backward_model.parameters(),
            lr=self.learning_rate
        )
        return [opt_forward, opt_backward]
        
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        cond: Optional[Dict] = None,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """采样过程"""
        shape = (batch_size, self.forward_model.in_channels,
                self.forward_model.image_size, self.forward_model.image_size)
                
        # 从标准正态分布采样
        x = torch.randn(shape, device=self.device)
        
        intermediates = []
        for i in tqdm(range(self.num_timesteps-1, -1, -1), desc="Sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # 使用EMA模型预测
            with self.ema_scope("Sampling"):
                if self.reparam_type == "flow":
                    pred = self.forward_model(x, t, cond)
                    gamma_t = self.gammas[t].view(-1, 1, 1, 1)
                    x = x + torch.sqrt(gamma_t) * pred
                else:  # term
                    x = self.forward_model(torch.cat([x, torch.zeros_like(x)], dim=1), t, cond)
                
            # 添加噪声
            if i > 0:
                noise = torch.randn_like(x)
                gamma_t = self.gammas[i-1].view(-1, 1, 1, 1)
                x = x + torch.sqrt(gamma_t) * noise
                
            if return_intermediates:
                intermediates.append(x.clone())
                
        if return_intermediates:
            return x, intermediates
        return x
        
    def encode_text(self, text):
        """使用CLIP编码文本
        
        Args:
            text: 文本描述
            
        Returns:
            文本的CLIP嵌入
        """
        if self.cond_stage_model is None:
            raise ValueError("CLIP encoder not initialized")
            
        return self.cond_stage_model.encode(text)
        
    def get_text_embeddings(self, batch):
        """从batch中获取文本嵌入
        
        Args:
            batch: 包含text_input的数据batch
            
        Returns:
            条件嵌入
        """
        if not exists(self.cond_stage_model):
            return None
            
        text = batch.get('text_input', None)
        if text is None:
            return None
            
        embeddings = self.encode_text(text)
        return embeddings
        
    def compute_clip_loss(self, x0, x1, text_embeddings=None):
        """计算CLIP相似度损失
        
        Args:
            x0: 源图像
            x1: 目标图像
            text_embeddings: 文本嵌入(可选)
            
        Returns:
            CLIP损失
        """
        if not exists(self.cond_stage_model):
            return torch.tensor(0.0, device=self.device)
            
        # 获取图像特征
        img_feat0 = self.cond_stage_model.encode_image(x0)
        img_feat1 = self.cond_stage_model.encode_image(x1)
        
        # 计算图像-图像相似度
        img_loss = 1 - torch.cosine_similarity(img_feat0, img_feat1, dim=-1).mean()
        
        # 如果有文本嵌入,计算图像-文本相似度
        if exists(text_embeddings):
            txt_sim0 = torch.cosine_similarity(img_feat0, text_embeddings, dim=-1)
            txt_sim1 = torch.cosine_similarity(img_feat1, text_embeddings, dim=-1)
            txt_loss = (1 - txt_sim1).mean() - (1 - txt_sim0).mean()
            return img_loss + txt_loss
            
        return img_loss