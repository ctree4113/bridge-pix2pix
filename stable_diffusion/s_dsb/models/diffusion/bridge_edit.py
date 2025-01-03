import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
from tqdm import tqdm

from .bridge import SDSB
from ...modules.encoders.modules import FrozenCLIPEmbedder
from ...util import exists, default

class SDSBEdit(SDSB):
    """S-DSB的指令化图像编辑实现"""
    
    def __init__(
        self,
        unet_config: Dict,
        clip_config: Dict,
        guidance_scale: float = 5.0,
        edit_threshold: float = 0.8,
        edit_strategy: str = "bidirectional",
        progressive_steps: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(
            unet_config=unet_config,
            clip_config=clip_config,
            **kwargs
        )
        
        self.guidance_scale = guidance_scale
        self.edit_threshold = edit_threshold
        self.edit_strategy = edit_strategy
        self.progressive_steps = progressive_steps or []
        
    def edit_image(self, image, edit_info):
        """编辑图像"""
        # 构建条件字典
        cond = {
            "text": edit_info,
            "c_concat": []
        }
        
        # 获取文本嵌入
        text_embeddings = self.get_learned_conditioning(cond)
        
        # 计算实际步数
        actual_steps = self.diffusion.core.num_timesteps
        
        # 前向编辑
        x_t = self.forward_edit(image, text_embeddings, actual_steps//2)
        
        # 反向采样
        edited = self.diffusion.sample(
            x_t,
            text_embeddings
        )
        
        return edited
        
    def forward_edit(
        self,
        image: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """前向编辑过程"""
        # 初始化噪声
        noise = torch.randn_like(image)
        
        # 生成时间步
        t = torch.ones((image.shape[0],), device=image.device) * num_steps
        t = t.long()
        
        # 前向过程
        x = self.diffusion.forward_process(
            x=image,
            t=t,
            noise=noise
        )
        
        return x
        
    def backward_edit(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """反向编辑过程"""
        with torch.no_grad():
            for i in reversed(range(num_steps)):
                t = torch.full((x.shape[0],), i, device=x.device)
                x = self.diffusion.backward_process(
                    x,
                    t,
                    {"text": text_embeddings}
                )
        return x
        
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        # 获取输入
        x = batch["edited"]  # 目标图像
        edit_info = batch["edit"]["c_crossattn"]  # 从 edit 字典中获取文本指令
        
        # 构建条件字典
        cond = {
            "text": edit_info,  # 文本条件
            "c_concat": batch["edit"].get("c_concat", [])  # 空间条件,如果存在的话
        }
        
        # 获取编码后的条件
        c = self.get_learned_conditioning(cond)
        
        # 计算损失
        t = torch.randint(0, self.diffusion.core.num_timesteps, (x.shape[0],), device=self.device)
        loss = self.loss_fn(
            self.diffusion,
            x,
            t,
            c
        )
        
        # 计算编辑质量
        if batch_idx % 100 == 0:
            with torch.no_grad():
                edited = self.edit_image(x, edit_info)
                quality_score = self.evaluate_edit_quality(x, edited, c)
                self.log("train/edit_quality", quality_score)
                
        self.log("train/loss", loss)
        return loss
        
    def evaluate_edit_quality(
        self,
        original: torch.Tensor,
        edited: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """评估编辑质量"""
        # 计算图像特征
        with torch.no_grad():
            orig_feat = self.diffusion.clip.encode_image(original)
            edit_feat = self.diffusion.clip.encode_image(edited)
            
        # 计算相似度
        edit_sim = F.cosine_similarity(edit_feat, text_embeddings)
        preserve_sim = F.cosine_similarity(edit_feat, orig_feat)
        
        # 综合评分
        quality = edit_sim * self.edit_threshold + preserve_sim * (1 - self.edit_threshold)
        
        return quality.mean()
        
    def get_learned_conditioning(self, cond):
        """处理条件信息,与 ddpm_edit.py 保持一致"""
        if isinstance(cond, dict):
            # 处理混合条件
            c_concat = cond.get("c_concat", [])
            c_crossattn = cond.get("text", "")  # 使用 "text" 作为键名
            
            # 处理空间条件
            if len(c_concat) > 0:
                c_concat = c_concat[0]  # 取第一个元素
            else:
                c_concat = None
            
            # 处理文本条件
            if c_crossattn:
                if isinstance(c_crossattn, list):
                    c_crossattn = c_crossattn[0]  # 如果是列表,取第一个元素
                if not isinstance(c_crossattn, str):
                    c_crossattn = str(c_crossattn)  # 确保是字符串类型
            else:
                c_crossattn = ""
            
            # 编码文本条件
            text_embed = self.diffusion.clip.encode(c_crossattn)
            
            return {
                "c_concat": [c_concat] if c_concat is not None else [],
                "text": text_embed  # 返回编码后的文本嵌入
            }
        else:
            # 处理单一条件
            if not isinstance(cond, str):
                cond = str(cond)  # 确保是字符串类型
            text_embed = self.diffusion.clip.encode(cond)
            return {"text": text_embed}