import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
from typing import Optional, Union, List, Tuple

class AbstractEncoder(nn.Module):
    """编码器基类"""
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(nn.Module):
    """冻结的CLIP编码器"""
    
    def __init__(
        self,
        version: str = "openai/clip-vit-large-patch14",
        freeze: bool = True,
        layer: str = "last"
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        
        if freeze:
            self.freeze()
            
        self.layer = layer
        
    def freeze(self):
        """冻结模型参数"""
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """前向传播"""
        if isinstance(text, str):
            text = [text]
            
        # 分词
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.transformer.device)
        
        # 获取文本特征
        outputs = self.transformer(**tokens)
        
        # 根据指定层返回特征
        if self.layer == "last":
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs.hidden_states[int(self.layer)]
            
        return embeddings

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            文本特征 [B, S, D]
        """
        return self(text)

class FrozenClipImageEmbedder(nn.Module):
    """使用CLIP作为图像编码器
    
    用于S-DSB的图像特征提取,支持图像条件。
    """
    def __init__(
        self,
        model: str = "ViT-L/14",
        device: str = 'cuda',
        antialias: bool = False,
        normalize: bool = True
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=False)
        self.antialias = antialias
        self.normalize = normalize
        self.device = device

        # 注册归一化参数
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]))
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        
        # 冻结模型
        self.freeze()

    def freeze(self):
        """冻结模型参数"""
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """预处理图像
        
        Args:
            x: 输入图像 [-1,1]
            
        Returns:
            预处理后的图像
        """
        # 缩放到224x224
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias
        )
        
        # 归一化到[0,1]
        x = (x + 1.) / 2.
        
        # CLIP归一化
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像 [-1,1]
            
        Returns:
            图像特征
        """
        # 预处理
        x = self.preprocess(x)
        
        # 编码
        features = self.model.encode_image(x)
        
        # 可选归一化
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像
        
        Args:
            x: 输入图像 [-1,1]
            
        Returns:
            图像特征
        """
        return self(x)

class SpatialRescaler(nn.Module):
    """空间缩放模块
    
    用于调整特征图的空间分辨率。
    """
    def __init__(
        self,
        n_stages: int = 1,
        method: str = 'bilinear',
        multiplier: float = 0.5,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        
        # 可选的通道映射
        self.remap_output = out_channels is not None
        if self.remap_output:
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            缩放后的特征图
        """
        # 多阶段缩放
        for _ in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        # 可选的通道映射
        if self.remap_output:
            x = self.channel_mapper(x)
            
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码接口
        
        Args:
            x: 输入特征图
            
        Returns:
            缩放后的特征图
        """
        return self(x)