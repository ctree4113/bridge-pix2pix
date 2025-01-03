import torch
import torch.nn as nn
from einops import rearrange

class SpatialTransformer(nn.Module):
    """
    空间 Transformer 模块
    """
    def __init__(
        self, 
        in_channels: int,
        n_heads: int, 
        d_head: int,
        depth: int = 1,
        context_dim: int = None
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels)
        
        self.proj_in = nn.Conv2d(in_channels,
                                inner_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim)
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(inner_dim,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # 注意力计算
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, context_dim: int = None):
        super().__init__()
        self.attn1 = CrossAttention(dim, dim, n_heads, d_head)  # self attention
        self.norm1 = nn.LayerNorm(dim)
        if context_dim is not None:
            self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)  # cross attention
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.attn2 = None
            self.norm2 = None
        self.ff = FeedForward(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        if self.attn2 is not None:
            x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        h = self.heads
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) 