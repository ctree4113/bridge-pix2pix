import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import List, Optional

from ...util import checkpoint, conv_nd, linear, zero_module, normalization
from ..attention import SpatialTransformer

def exists(x):
    return x is not None

def get_timestep_embedding(timesteps, embedding_dim):
    """构建时间步的正弦嵌入"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class ResBlock(nn.Module):
    """S-DSB专用的残差块"""
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """S-DSB专用的注意力块"""
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_checkpoint = use_checkpoint

        if num_head_channels == -1:
            self.num_heads = 1
            self.num_head_channels = channels
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """QKV注意力机制"""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
        dropout: float = 0.0,
        temb_channels: Optional[int] = None
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # 根据通道数选择合适的组数
        if use_norm:
            if out_channels < 8:
                self.norm = nn.GroupNorm(1, out_channels)  # 如果通道数小于8,使用单组
            else:
                self.norm = nn.GroupNorm(8, out_channels)
        else:
            self.norm = nn.Identity()
            
        self.act = nn.SiLU() if use_act else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 时间嵌入投影
        if exists(temb_channels):
            self.temb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(temb_channels, out_channels)
            )
        else:
            self.temb_proj = None
        
    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        h = self.conv(x)
        h = self.norm(h)
        
        # 添加时间嵌入
        if exists(self.temb_proj) and exists(temb):
            temb = self.temb_proj(temb)[:, :, None, None]
            h = h + temb
            
        h = self.act(h)
        h = self.dropout(h)
        return h

class SDSBUNet(nn.Module):
    """S-DSB的U-Net实现"""
    
    def __init__(
        self,
        model_channels: int = 96,
        attention_resolutions: List[int] = [4],
        channel_mult: List[int] = [1, 2, 3],
        num_heads: int = 4,
        context_dim: int = 256,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1
    ):
        super().__init__()
        
        # 保存配置
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 编码器
        self.encoder = nn.ModuleList()
        input_channels = 3  # 从原始图像的通道数开始
        for mult in channel_mult:
            output_channels = model_channels * mult
            self.encoder.append(
                ConvBlock(
                    input_channels,
                    output_channels,
                    temb_channels=time_embed_dim
                )
            )
            input_channels = output_channels  # 更新下一层的输入通道数
            
        # 注意力层
        self.attention = nn.ModuleList([
            EfficientAttention(
                model_channels * mult,
                num_heads=num_heads,
                context_dim=context_dim
            )
            for mult in channel_mult
            if mult in attention_resolutions
        ])
        
        # 解码器
        self.decoder = nn.ModuleList()
        reversed_mult = list(reversed(channel_mult))
        input_channels = model_channels * reversed_mult[0]  # 从最后一层编码器的输出开始
        
        for i in range(len(reversed_mult)):
            # 计算跳跃连接的通道数
            skip_channels = model_channels * reversed_mult[i]
            
            # 计算输出通道数
            if i == len(reversed_mult) - 1:  # 最后一层
                out_channels = 3  # 输出RGB图像
            else:
                out_channels = model_channels * reversed_mult[i+1]
                
            # 总输入通道数 = 当前特征通道数 + 跳跃连接通道数
            total_in_channels = input_channels + skip_channels
                
            self.decoder.append(
                ConvBlock(
                    total_in_channels,
                    out_channels,
                    temb_channels=time_embed_dim
                )
            )
            
            input_channels = out_channels  # 更新下一层的输入通道数
            
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        # 时间嵌入
        t_emb = get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # 编码
        h = []
        for enc in self.encoder:
            x = enc(x, t_emb)
            h.append(x)
            
        # 注意力
        for attn in self.attention:
            x = attn(x, context=context)
            
        # 解码
        for dec in self.decoder:
            x = dec(torch.cat([x, h.pop()], dim=1), t_emb)
            
        return x

class TimestepEmbedSequential(nn.Sequential):
    """支持时间步和上下文的Sequential"""
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """上采样模块"""
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """下采样模块"""
    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(
                dims, channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        # 减少中间特征维度
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        # 使用memory efficient attention
        self.attention = MemoryEfficientAttention()

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 重塑张量以减少内存使用
        q = q.reshape(-1, h, q.shape[-1] // h)
        k = k.reshape(-1, h, k.shape[-1] // h)
        v = v.reshape(-1, h, v.shape[-1] // h)

        # 使用memory efficient attention
        out = self.attention(q, k, v)
        
        # 重塑回原始维度
        out = out.reshape(x.shape[0], -1, x.shape[-1])
        return self.to_out(out)

class MemoryEfficientAttention(nn.Module):
    """内存高效的attention实现"""
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):
        # 分块计算attention
        chunk_size = 128
        chunks = []
        
        for i in range(0, q.shape[0], chunk_size):
            q_chunk = q[i:i+chunk_size]
            k_chunk = k[i:i+chunk_size] 
            v_chunk = v[i:i+chunk_size]
            
            # 计算attention scores
            scores = torch.bmm(q_chunk, k_chunk.transpose(-2, -1))
            scores = F.softmax(scores, dim=-1)
            
            # 计算attention输出
            chunk_out = torch.bmm(scores, v_chunk)
            chunks.append(chunk_out)
            
        return torch.cat(chunks, dim=0)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        
        # 获取输入维度
        B, C, H, W = h_.shape
        
        # 计算q, k, v
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # 重塑维度以进行attention计算
        q = q.reshape(B, C, -1)  # B, C, HW
        k = k.reshape(B, C, -1)  # B, C, HW
        v = v.reshape(B, C, -1)  # B, C, HW
        
        # 分块计算attention
        chunk_size = min(128, H*W)  # 选择合适的块大小
        attn_outputs = []
        
        for i in range(0, H*W, chunk_size):
            # 获取当前块
            q_chunk = q[:, :, i:i+chunk_size]  # B, C, chunk_size
            
            # 计算attention scores (使用半精度以节省内存)
            with torch.cuda.amp.autocast():
                # 计算当前块的attention scores
                attn = torch.bmm(q_chunk.transpose(1, 2), k)  # B, chunk_size, HW
                attn = attn * (int(C) ** (-0.5))
                
                # 使用chunk计算softmax
                attn = torch.softmax(attn, dim=-1)
                
                # 计算attention输出
                chunk_out = torch.bmm(attn, v.transpose(1, 2))  # B, chunk_size, C
            
            attn_outputs.append(chunk_out)
        
        # 合并所有块的输出
        h_ = torch.cat(attn_outputs, dim=1)  # B, HW, C
        
        # 重塑回原始维度
        h_ = h_.transpose(1, 2).reshape(B, C, H, W)
        
        # 投影输出
        h_ = self.proj_out(h_)
        
        return x + h_

def memory_efficient_attention(q, k, v, scale=None):
    """内存高效的attention实现"""
    B, H, N, C = q.shape
    
    if scale is None:
        scale = C ** -0.5
        
    chunk_size = min(128, N)
    attn_outputs = []
    
    # 分块计算attention
    for i in range(0, N, chunk_size):
        q_chunk = q[:, :, i:i+chunk_size]
        
        # 使用半精度计算
        with torch.cuda.amp.autocast():
            # 计算attention scores
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            
            # 使用log_softmax和exp来避免数值不稳定
            scores = torch.log_softmax(scores, dim=-1)
            scores = torch.exp(scores)
            
            # 计算attention输出
            chunk_out = torch.matmul(scores, v)
            
        attn_outputs.append(chunk_out)
        
    # 合并所有块的输出
    return torch.cat(attn_outputs, dim=2)

class EfficientAttention(nn.Module):
    """内存高效的注意力机制"""
    
    def __init__(self, dim: int, num_heads: int = 8, context_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = dim
        
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        
        if exists(context_dim):
            self.context_proj = nn.Linear(context_dim, inner_dim)
            
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        if exists(context):
            context = self.context_proj(context)
            k = torch.cat([k, context], dim=1)
            v = torch.cat([v, context], dim=1)
            
        # 分头
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.num_heads, -1).transpose(1, 2)
        
        # 高效注意力计算
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # 合并头
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return out