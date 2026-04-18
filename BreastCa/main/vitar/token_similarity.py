# In our current model, the token similarities are handle through the attention mechanism. 

import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from zeta import (
    MultiQueryAttention,
    FeedForward,
    patch_img,
)

# Original Method
class PooledAttentionSimilarity(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor) -> Tensor:
        # Original method logic
        k = self.proj(x)
        v = self.proj(x)
        
        # Average pool queries
        q = x.transpose(1, 2)  # [B, D, S]
        q = F.adaptive_avg_pool1d(q, q.size(-1) // 2)  # Reduce sequence length
        q = q.transpose(1, 2)  # [B, S/2, D]
        
        # Project queries
        q = self.proj(q)
        
        return q, k, v

# 1st alternative. Euclidean similarity
# Computes pairwise Euclidean distances between tokens
# Converts distances to similarities using negative exponential, Lower distance = higher similarity
class EuclideanSimilarity(nn.Module):
    def __init__(self, dim: int, heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project features
        x = self.proj(x)  # [B, S, D]
        
        # Compute pairwise distances
        x_expanded = x.unsqueeze(2)  # [B, S, 1, D]
        y_expanded = x.unsqueeze(1)  # [B, 1, S, D]
        
        # Compute Euclidean distance
        distances = torch.sqrt(torch.sum((x_expanded - y_expanded) ** 2, dim=-1))  # [B, S, S]
        
        # Convert distances to similarities (negative exponential)
        similarities = torch.exp(-distances * self.temperature)
        
        return similarities

# 2nd alternative. Cosine similarity
# Uses normalized dot product between tokens
class CosineSimilarity(nn.Module):
    def __init__(self, dim: int, heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project features
        x = self.proj(x)  # [B, S, D]
        
        # Normalize features
        x_norm = F.normalize(x, p=2, dim=-1)
        
        # Compute cosine similarity
        similarities = torch.matmul(x_norm, x_norm.transpose(-2, -1)) * self.temperature
        
        return similarities

# 3rd. Attention score similarity
# Uses scaled dot-product attention mechanism
class AttentionSimilarity(nn.Module):
    def __init__(self, dim: int, heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.temperature = temperature
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Project to queries and keys
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        similarities = torch.softmax(scores * self.temperature, dim=-1)
        
        # Average across heads
        similarities = similarities.mean(dim=1)
        
        return similarities

# 4th. Semantic Simility
# Projects token into a learned 
class SemanticSimilarity(nn.Module):
    def __init__(self, dim: int, semantic_dim: int = 64, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.semantic_dim = semantic_dim
        self.heads = heads
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(dim, semantic_dim * 2),
            nn.GELU(),
            nn.Linear(semantic_dim * 2, semantic_dim)
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        semantic_tokens = self.semantic_proj(x)
        semantic_tokens = F.normalize(semantic_tokens, p=2, dim=-1)
        
        similarities = torch.matmul(semantic_tokens, semantic_tokens.transpose(-2, -1))
        
        k = x.shape[1] // 2
        topk_sim, indices = torch.topk(similarities, k, dim=-1)
        
        q = torch.gather(x, 1, indices[:, :, 0].unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        return q, x, x
    
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

class AdaptiveTokenMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        similarity_method: str = 'pooled_attention',
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        
        # Choose similarity method
        if similarity_method == 'pooled_attention':
            self.similarity = PooledAttentionSimilarity(dim, heads)
        elif similarity_method == 'euclidean':
            self.similarity = EuclideanSimilarity(dim, heads, temperature)
        elif similarity_method == 'cosine':
            self.similarity = CosineSimilarity(dim, heads, temperature)
        elif similarity_method == 'semantic':
            self.similarity = SemanticSimilarity(dim, heads)
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        # MultiQuery Attention for token processing
        self.attn = MultiQueryAttention(dim, heads)
        
        # FFN and norms
        self.ffn = FeedForward(dim, dim, 4, swish=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        # Compute similarities and get q, k, v
        q, k, v = self.similarity(x)
        
        # Apply attention
        attn_out, _, _ = self.attn(q, k, v)
        attn_out = self.dropout(attn_out)
        
        # FFN
        out = self.norm1(attn_out)
        out = self.ffn(out)
        out = self.dropout(out)
        out = self.norm2(out)
        
        return out

class Vitar(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 12,
        similarity_method: str = 'pooled_attention',
        dropout: float = 0.1,
        depth: int = 12,
        patch_size: int = 16,
        image_size: int = 224,
        channels: int = 3,
        num_classes: int = 1000,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm(dim),
        )
        
        # Position embedding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Adaptive token merger
        self.token_merger = AdaptiveTokenMerger(
            dim, 
            heads,
            similarity_method,
            dropout,
            temperature
        )
        
        # Transformer blocks
        self.transformer_blocks = TransformerBlocks(
            dim,
            heads,
            dropout,
            depth,
        )
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        # Patch embedding
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add position embeddings
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Prepend CLS token
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Adaptive token merging
        x = self.token_merger(x)
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        
        # Classification from CLS token
        x = x[:, 0]  # Take CLS token
        x = self.norm(x)
        x = self.head(x)
        
        return x

