from torch import nn, Tensor
from zeta import (
    MultiQueryAttention,
    FeedForward,
    patch_img,
)
import torch.nn.functional as F

class GridAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        reduction_factor: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.reduction_factor = reduction_factor

        # Separate projections for better expressiveness
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        target_len = seq_len // self.reduction_factor
        
        # Apply layer norm
        x = self.norm(x)
        
        # Project to Q, K, V
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Adaptive pooling for queries to reduce tokens
        # Properly reshape and transpose for pooling
        q = self.q_proj(x)
        q = q.transpose(1, 2)  # (B, D, S)
        q = F.adaptive_avg_pool1d(q, target_len)  # (B, D, target_len)
        q = q.transpose(1, 2)  # (B, target_len, D)

        # Use MultiQueryAttention with reduced sequence length
        out, _, _ = MultiQueryAttention(self.dim, self.heads)(q, k, v)
        return self.out_proj(out)

class AdaptiveTokenMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        reduction_factor: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        
        # Grid attention with reduction factor
        self.attn = GridAttention(dim, heads, reduction_factor)
        
        # FFN with dropout
        self.ffn = FeedForward(dim, dim, 4, swish=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # First normalize
        x = self.norm1(x)
        
        # Grid Attention
        grid = self.attn(x)
        grid = self.dropout(grid)
        
        # Second normalize
        grid = self.norm2(grid)
        
        # FFN
        ffn = self.ffn(grid)
        ffn = self.dropout(ffn)
        
        return ffn

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int = 4,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # ffn
        self.ffn = FeedForward(
            dim,
            dim,
            ffn_dim,
            swish=True,
        )

        # Attention
        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Norms
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.norm(x)

        # Multi-Query Attention
        out, _, _ = self.attn(x)

        # Add and Norm
        out = self.norm(out) + residual

        # 2nd path
        residual_two = out

        # FFN
        ffd = self.norm(self.ffn(out))

        return ffd + residual_two

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        depth: int = 9,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth

        # transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Loop through the blocks
        for block in self.blocks:
            x = block(x)

        return x

class Vitar(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        depth: int = 9,
        patch_size: int = 16,
        image_size: int = 224,
        channels: int = 3,
        ffn_dim: int = 4,
        num_classes: int = 1000,
        reduction_factor: int = 2,  # Added parameter for token reduction
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        
        # Transformer Blocks
        self.transformer_blocks = TransformerBlocks(
            dim,
            heads,
            dropout,
            depth,
        )
        
        # Norm and classification head
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, x) -> Tensor:
        # Embed the image -> (B, S, D)
        x = patch_img(x, self.patch_size)
        
        b, s, d = x.shape
        
        # Adaptive token merger with specified reduction
        out = AdaptiveTokenMerger(
            d, 
            self.heads, 
            self.dropout, 
            reduction_factor=2
        )(x)
        
        # Transformer Blocks
        transformed = self.transformer_blocks(out)
        
        # Classification head
        x = transformed.mean(dim=1)
        x = self.to_latent(x)
        
        return self.linear_head(x)