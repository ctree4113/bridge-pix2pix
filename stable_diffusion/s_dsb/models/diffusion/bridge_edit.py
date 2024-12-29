import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any, Callable
from tqdm import tqdm

from .bridge import SDSB
from .samplers import SDSBSampler
from ...modules.encoders.modules import FrozenCLIPEmbedder
from ...util import exists, default

class SDSBEdit(SDSB):
    """S-DSB的指令化图像编辑实现
    
    基于S-DSB的文本条件生成模型,专注于指令化图像编辑任务。
    支持前向/反向/双向编辑策略,以及渐进式编辑。
    """
    
    def __init__(
        self,
        unet_config: Dict,
        clip_config: Optional[Dict] = None,
        num_timesteps: int = 1000,
        gamma_schedule: str = "linear",
        gamma_min: float = 1e-4,
        gamma_max: float = 0.1,
        reparam_type: str = "flow",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        learning_rate: float = 1e-4,
        guidance_scale: float = 7.5,
        edit_threshold: float = 0.8,
        edit_strategy: str = "forward",  # "forward", "backward", "bidirectional"
        progressive_steps: Optional[List[int]] = None,
        memory_efficient: bool = False,
        **kwargs
    ):
        super().__init__(
            unet_config=unet_config,
            clip_config=clip_config,
            num_timesteps=num_timesteps,
            gamma_schedule=gamma_schedule,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            reparam_type=reparam_type,
            use_ema=use_ema,
            ema_decay=ema_decay,
            learning_rate=learning_rate,
            **kwargs
        )
        
        # 编辑相关参数
        self.guidance_scale = guidance_scale
        self.edit_threshold = edit_threshold
        self.edit_strategy = edit_strategy
        self.progressive_steps = progressive_steps or []
        self.memory_efficient = memory_efficient
        
        # 初始化采样器
        self.sampler = SDSBSampler(
            num_timesteps=num_timesteps,
            gamma_scheduler=self.gamma_scheduler,
            reparam_type=reparam_type,
            use_ema=use_ema
        )
        
    def get_text_embeddings(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """获取文本嵌入"""
        if isinstance(prompt, str):
            prompt = [prompt]
            
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
            
        text_embeddings = self.cond_stage_model(prompt)
        uncond_embeddings = self.cond_stage_model(negative_prompt)
        
        return torch.cat([uncond_embeddings, text_embeddings])
        
    def edit_step_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: float,
        **kwargs
    ) -> torch.Tensor:
        """前向编辑步骤"""
        # 复制输入
        latent_model_input = torch.cat([x_t] * 2)
        t_input = torch.cat([t.unsqueeze(0)] * 2)
        
        # 模型预测
        with self.ema_scope("Forward Editing"):
            noise_pred = self.forward_model(
                latent_model_input,
                t_input,
                encoder_hidden_states=text_embeddings
            )
            
        # 分离预测结果
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # 分类器引导
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # 重参数化
        if self.reparam_type == "flow":
            gamma_t = self.gammas[t].view(-1, 1, 1, 1)
            x_t = x_t + torch.sqrt(gamma_t) * noise_pred
        else:  # term
            x_t = noise_pred
            
        return x_t.clamp(-1., 1.)
        
    def edit_step_backward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: float,
        **kwargs
    ) -> torch.Tensor:
        """反向编辑步骤"""
        # 复制输入
        latent_model_input = torch.cat([x_t] * 2)
        t_input = torch.cat([t.unsqueeze(0)] * 2)
        
        # 模型预测
        with self.ema_scope("Backward Editing"):
            noise_pred = self.backward_model(
                latent_model_input,
                t_input,
                encoder_hidden_states=text_embeddings
            )
            
        # 分离预测结果
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # 分类器引导
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # 重参数化
        if self.reparam_type == "flow":
            gamma_t = self.gammas[t].view(-1, 1, 1, 1)
            x_t = x_t + torch.sqrt(gamma_t) * noise_pred
        else:  # term
            x_t = noise_pred
            
        return x_t.clamp(-1., 1.)
        
    def combine_results(
        self,
        forward_result: torch.Tensor,
        backward_result: torch.Tensor,
        weight: float = 0.5
    ) -> torch.Tensor:
        """组合前向和反向结果"""
        return weight * forward_result + (1 - weight) * backward_result
        
    def progressive_edit(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        steps: List[int],
        **kwargs
    ) -> List[torch.Tensor]:
        """渐进式编辑
        
        通过多个阶段逐步编辑图像,每个阶段使用不同的编辑强度。
        """
        results = []
        current_image = image
        
        for step in steps:
            kwargs["edit_threshold"] = step / self.num_timesteps
            edited = self.edit(current_image, prompt, **kwargs)
            results.append(edited)
            current_image = edited
            
        return results
        
    def prepare_edit_latents(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        edit_threshold: float = 0.8,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备编辑潜在表示"""
        # 获取文本嵌入
        text_embeddings = self.get_text_embeddings(prompt, negative_prompt)
        
        # 添加噪声
        t_start = int(self.num_timesteps * edit_threshold)
        t = torch.full((image.shape[0],), t_start, device=self.device)
        noise = torch.randn_like(image)
        noisy_image = self.q_sample(image, t, noise)
        
        return noisy_image, text_embeddings
        
    @torch.no_grad()
    def edit(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        edit_threshold: Optional[float] = None,
        return_intermediates: bool = False,
        edit_strategy: Optional[str] = None,
        progressive: bool = False,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """指令化图像编辑"""
        # 使用默认参数
        guidance_scale = default(guidance_scale, self.guidance_scale)
        edit_threshold = default(edit_threshold, self.edit_threshold)
        edit_strategy = default(edit_strategy, self.edit_strategy)
        
        # 准备潜在表示
        noisy_image, text_embeddings = self.prepare_edit_latents(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            edit_threshold=edit_threshold,
            **kwargs
        )
        
        # 渐进式编辑
        if progressive and exists(self.progressive_steps):
            return self.progressive_edit(
                image=image,
                prompt=prompt,
                steps=self.progressive_steps,
                **kwargs
            )
            
        # 计算时间步
        t_start = int(self.num_timesteps * edit_threshold)
        timesteps = torch.linspace(t_start, 0, num_inference_steps, device=self.device)
        timesteps = timesteps.long()
        
        # 存储中间结果
        intermediates = []
        
        # 编辑过程
        for i, t in enumerate(tqdm(timesteps, desc="Editing")):
            # 根据策略选择编辑方法
            if edit_strategy == "forward":
                noisy_image = self.edit_step_forward(
                    noisy_image, t, text_embeddings, guidance_scale, **kwargs
                )
            elif edit_strategy == "backward":
                noisy_image = self.edit_step_backward(
                    noisy_image, t, text_embeddings, guidance_scale, **kwargs
                )
            else:  # bidirectional
                forward_result = self.edit_step_forward(
                    noisy_image, t, text_embeddings, guidance_scale, **kwargs
                )
                backward_result = self.edit_step_backward(
                    noisy_image, t, text_embeddings, guidance_scale, **kwargs
                )
                noisy_image = self.combine_results(forward_result, backward_result)
            
            # 添加噪声(如果不是最后一步)
            if i < len(timesteps) - 1:
                noise = torch.randn_like(noisy_image)
                next_gamma = self.gammas[timesteps[i+1]].view(-1, 1, 1, 1)
                noisy_image = noisy_image + torch.sqrt(next_gamma) * noise
                
            if return_intermediates:
                intermediates.append(noisy_image.clone())
                
            # 回调函数
            if exists(callback):
                callback(i, t, noisy_image)
                
        if return_intermediates:
            return noisy_image, intermediates
        return noisy_image
        
    def compute_edit_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """计算编辑损失"""
        # 基础重建损失
        base_loss = F.mse_loss(pred, target)
        
        # gamma加权
        gamma_t = self.gammas[t].view(-1, 1, 1, 1)
        weighted_loss = base_loss / torch.sqrt(gamma_t)
        
        return weighted_loss.mean()
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        """训练步骤"""
        loss = super().training_step(batch, batch_idx, optimizer_idx)
        
        # 添加编辑特定的损失
        if hasattr(batch, "edit_data"):
            edit_loss = self.compute_edit_loss(
                pred=batch["edit_pred"],
                target=batch["edit_target"],
                t=batch["t"]
            )
            loss = loss + edit_loss
            self.log("train/edit_loss", edit_loss)
            
        return loss
        
    def compute_edit_similarity(self, x_orig, x_edit, text_embeddings=None):
        """计算编辑前后的相似度
        
        Args:
            x_orig: 原始图像
            x_edit: 编辑后图像
            text_embeddings: 编辑指令的文本嵌入
            
        Returns:
            相似度分数
        """
        # 获取图像特征
        feat_orig = self.cond_stage_model.encode_image(x_orig)
        feat_edit = self.cond_stage_model.encode_image(x_edit)
        
        # 计算图像相似度
        img_sim = torch.cosine_similarity(feat_orig, feat_edit, dim=-1)
        
        if exists(text_embeddings):
            # 计算与编辑指令的相似度提升
            txt_sim_orig = torch.cosine_similarity(feat_orig, text_embeddings, dim=-1)
            txt_sim_edit = torch.cosine_similarity(feat_edit, text_embeddings, dim=-1)
            txt_improve = txt_sim_edit - txt_sim_orig
            
            return img_sim, txt_improve
            
        return img_sim
        
    def apply_clip_guidance(self, x, t, text_embeddings, guidance_scale):
        """应用CLIP引导
        
        Args:
            x: 当前状态
            t: 时间步
            text_embeddings: 文本嵌入
            guidance_scale: 引导强度
            
        Returns:
            引导后的状态
        """
        # 获取无条件和条件预测
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([torch.zeros_like(text_embeddings), text_embeddings])
        
        # 预测噪声
        e_t = self.forward_model(x_in, t_in, c_in).chunk(2)
        e_t_uncond, e_t_cond = e_t[0], e_t[1]
        
        # 应用分类器引导
        e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)
        
        return e_t
        
    def evaluate_edit_quality(self, x_orig, x_edit, text_embeddings):
        """评估编辑质量
        
        Args:
            x_orig: 原始图像
            x_edit: 编辑后图像
            text_embeddings: 编辑指令的文本嵌入
            
        Returns:
            质量评分和指标
        """
        # 计算相似度
        img_sim, txt_improve = self.compute_edit_similarity(
            x_orig, x_edit, text_embeddings
        )
        
        # 计算编辑强度
        edit_strength = 1 - img_sim.mean()
        
        # 计算指令遵循度
        instruction_alignment = txt_improve.mean()
        
        # 综合质量分数
        quality_score = instruction_alignment * (1 - abs(edit_strength - self.edit_threshold))
        
        metrics = {
            "edit_strength": edit_strength,
            "instruction_alignment": instruction_alignment,
            "quality_score": quality_score
        }
        
        return quality_score, metrics