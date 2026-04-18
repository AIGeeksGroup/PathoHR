# In our current model, the token similarities are handle through the attention mechanism. 

import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r


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
        self.temperature = temperature
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Project features
        projected = self.proj(x)  # [B, S, D]
        
        # Pool to get queries
        q = x.transpose(1, 2)  # [B, D, S]
        q = F.adaptive_avg_pool1d(q, q.size(-1) // 2)
        q = q.transpose(1, 2)  # [B, S/2, D]
        q = self.proj(q)  # [B, S/2, D]
        
        # Compute pairwise distances between q and projected
        q_expanded = q.unsqueeze(2)  # [B, S/2, 1, D]
        k_expanded = projected.unsqueeze(1)  # [B, 1, S, D]
        
        power = torch.sum((q_expanded - k_expanded) ** 2, dim=-1)
        print(power.shape)
        # Euclidean distance
        distances = torch.sqrt(power)  # [B, S/2, S]
        
        # Convert distances to similarities
        similarities = torch.exp(-distances * self.temperature)  # [B, S/2, S]
        
        # Apply similarities to get new k and v
        k = torch.bmm(similarities, projected)  # [B, S/2, D]
        v = k  # Use same values for v
        
        return q, k, v

# 2nd alternative. Cosine similarity
# Uses normalized dot product between tokens
class CosineSimilarity(nn.Module):
    def __init__(self, dim: int, heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Project features
        projected = self.proj(x)  # [B, S, D]
        
        # Pool to get queries
        q = x.transpose(1, 2)  # [B, D, S]
        q = F.adaptive_avg_pool1d(q, q.size(-1) // 2)
        q = q.transpose(1, 2)  # [B, S/2, D]
        q = self.proj(q)  # [B, S/2, D]
        
        # Normalize vectors for cosine similarity
        q_norm = F.normalize(q, p=2, dim=-1)  # [B, S/2, D]
        k_norm = F.normalize(projected, p=2, dim=-1)  # [B, S, D]
        
        # Compute cosine similarities
        similarities = torch.bmm(q_norm, k_norm.transpose(1, 2))  # [B, S/2, S]
        similarities = similarities * self.temperature
        
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Compute weighted keys and values
        k = torch.bmm(attention_weights, projected)  # [B, S/2, D]
        v = k  # Use same values for v
        
        return q, k, v

# 3rd. Attention score similarity
# Uses scaled dot-product attention mechanism
class AttentionSimilarity(nn.Module):
    def __init__(self, dim: int, heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.scale = dim ** -0.5
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Pool to get queries
        q_input = x.transpose(1, 2)  # [B, D, S]
        q_input = F.adaptive_avg_pool1d(q_input, q_input.size(-1) // 2)  # Reduce sequence length
        q_input = q_input.transpose(1, 2)  # [B, S/2, D]
        
        # Project q, k, v
        q = self.proj_q(q_input)  # [B, S/2, D]
        k = self.proj_k(x)  # [B, S, D]
        v = self.proj_v(x)  # [B, S, D]
        
        # Compute scaled dot-product attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, S/2, S]
        attn = F.softmax(attn * self.temperature, dim=-1)
        
        # Apply attention to values
        k_out = torch.bmm(attn, v)  # [B, S/2, D]
        
        return q, k_out, k_out

# 4th. Semantic Simility
# Projects token into a learned 
class SemanticSimilarity(nn.Module):
    def __init__(self, dim: int, semantic_dim: int = 64, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.semantic_dim = semantic_dim
        self.proj = nn.Linear(dim, dim)
        
        # Semantic projection layers
        self.semantic_proj_q = nn.Sequential(
            nn.Linear(dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, dim)
        )
        
        self.semantic_proj_k = nn.Sequential(
            nn.Linear(dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, dim)
        )
        
        self.semantic_proj_v = nn.Sequential(
            nn.Linear(dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, dim)
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Pool to get queries
        q_input = x.transpose(1, 2)  # [B, D, S]
        q_input = F.adaptive_avg_pool1d(q_input, q_input.size(-1) // 2)  # Reduce sequence length
        q_input = q_input.transpose(1, 2)  # [B, S/2, D]
        
        # Project through semantic spaces
        q = self.semantic_proj_q(q_input)  # [B, S/2, D]
        k = self.semantic_proj_k(x)  # [B, S, D]
        v = self.semantic_proj_v(x)  # [B, S, D]
        
        # Compute semantic similarities
        similarities = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)  # [B, S/2, S]
        attn_weights = F.softmax(similarities, dim=-1)
        
        # Apply attention to get final k and v
        k_out = torch.bmm(attn_weights, v)  # [B, S/2, D]
        
        return q, k_out, k_out
class ToMeAttention(torch.nn.MultiheadAttention):
    """
    Modifications:
    - Apply proportional attention
    - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        # Return k as well here
        # return x, k.mean(1)
        return x
    
class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def __init__(self, dim, heads):
        super(ToMeBlock, self).__init__(dim, heads)
        self._tome_info = {
        "r": 16,
        "size": None,
        "source": None,
        "trace_source": False,
        "prop_attn": True,
        "class_token": False,
        "distill_token": False,
    }
        self.proj = nn.Linear(dim, dim)

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        # print(attn_size)
        # x_attn, metric = self.attn(self.norm1(x), attn_size)
        x_attn = self.attn(self.norm1(x))
        # metric = self.attn(attn_size)
        metric = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)

        #r = self._tome_info["r"].pop(0)
        r = self._tome_info["r"]
        r = 16
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))

        # Original method logic
        k = self.proj(x)
        v = self.proj(x)
        
        # Average pool queries
        q = x.transpose(1, 2)  # [B, D, S]
        q = F.adaptive_avg_pool1d(q, q.size(-1) // 2)  # Reduce sequence length
        q = q.transpose(1, 2)  # [B, S/2, D]
        
        # Project queries
        q = self.proj(q)
        
        
        # y = self.mlp(y)
        
        return q, k, v


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
        
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.norm(x)
        

        # Multi-Query Attention
        out, _, _ = self.attn(x)

        # Add and Norm
        # out = self.norm(out) + residual
        
        # remove residual blocks
        out = self.norm(out)

        # 2nd path
        residual_two = out

        # FFN
        ffd = self.norm(self.ffn(out))

        # return ffd + residual_two
        return ffd

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
        similarity_method: str = 'tome',
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        
        # Choose similarity method
        if similarity_method == 'tome':
            self.similarity = ToMeBlock(dim, heads)
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        # MultiQuery Attention for token processing
        self.attn = MultiQueryAttention(dim, heads)
        
        # FFN
        self.ffn = FeedForward(dim, dim, 4, swish=True)
        
        self.tome_attn = ToMeAttention(dim, heads)
        
    def forward(self, x: Tensor) -> Tensor:
        
        # Compute similarities and get q, k, v
        # x = self.tome_attn(x)
        
        q , k ,v = self.similarity(x)
        
        # Pool k and v to match q's sequence length
        k = k.transpose(1, 2)  # [B, D, S]
        v = v.transpose(1, 2)  # [B, D, S]
        k = F.adaptive_avg_pool1d(k, q.size(1))  # [B, D, S/2]
        v = F.adaptive_avg_pool1d(v, q.size(1))  # [B, D, S/2]
        k = k.transpose(1, 2)  # [B, S/2, D]
        v = v.transpose(1, 2)  # [B, S/2, D]
        
        # Add the tensors as in original paper
        combined = q + k + v
        
        # Apply attention
        out = self.attn(combined)[0]
        
        # FFN
        out = self.ffn(out)
        
        return out
    
class FeatureVitar(nn.Module):
    def __init__(
        self,
        ori_feat:int,
        similarity_method: str,
        dim: int = 1024,
        heads: int = 8,
        dropout: float = 0.1,
        depth: int = 9,
        num_classes: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.num_classes = num_classes

        # Adaptive token merger with chosen similarity method
        self.token_merger = AdaptiveTokenMerger(
            dim=dim,
            heads=heads,
            similarity_method=similarity_method,
            dropout=dropout,
            temperature=temperature
        )

        # Transformer blocks
        self.transformer_blocks = TransformerBlocks(
            dim=dim,
            heads=heads,
            dropout=dropout,
            depth=depth
        )

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.ori = ori_feat

    def forward(self, features: Tensor) -> Tensor:
        # Token merger
        merged = self.token_merger(features)
        print(merged.shape) # [105, 256, 1024]
        # Transform
        transformed = self.transformer_blocks(merged)
        print(transformed.shape) # [105, 256, 1024]
        
        # Global average pooling
        pooled = transformed.mean(dim=1)
        print(pooled.shape) # [105, 1024]
        
        # Normalize and classify
        normalized = self.norm(pooled) # [105, 1024]
        norm_first = int(list(normalized.shape)[0])

        output = nn.Linear(norm_first,self.ori)(normalized.permute(1,0)).permute(1,0)

        # output = self.head(normalized)
        print('output_shape:', output.shape) # [53875, 1024]

        return output

def process_features(feature_path: str, similarity_method: str, sequence_length: int = 512):
    print("Loading features...")
    features = torch.load(feature_path)
    print(f"Loaded features shape: {features.shape}") # [n, 1024]
    ori_feat = int(list(features.shape)[0])
    print("Original first shape is", ori_feat)
    
    # Reshape features into sequences
    total_patches = features.size(0)
    dim = features.size(1)
    
    # Calculate number of complete sequences
    num_sequences = total_patches // sequence_length
    if num_sequences == 0:
        sequence_length = total_patches
        num_sequences = 1
    
    print(f"Reshaping into {num_sequences} sequences of length {sequence_length}")
    features = features[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, dim)
    print(f"Reshaped features shape: {features.shape}")
    
    # Initialize model with specified similarity method
    model = FeatureVitar(
        dim=dim,
        heads=8,
        similarity_method=similarity_method,
        depth=9,
        num_classes=2,
        ori_feat=ori_feat
    )
    
    # Process features
    print(f"\nProcessing features through model using {similarity_method} similarity...")
    with torch.no_grad():
        outputs = model(features)

    print("\nProcessing complete!")
    return outputs

import os
# Example usage
if __name__ == "__main__":
    path = r"/date2/zhang_h/D/BreastCa/results/clam_features/16mul16pt" # feature path
    save_path = r"/date2/zhang_h/D/BreastCa/data/modified_features"
    similarity_method = 'tome'  # Try different methods: 'pooled_attention', 'euclidean', 'cosine', 'semantic','attention_score' 'tome'
    dis_save_path = os.path.join(save_path, similarity_method+'without_res_new')
    os.makedirs(dis_save_path, exist_ok=True)
    for i, file in enumerate(os.listdir(path)):
        print("No.",i)
        save = os.path.join(dis_save_path, file)
        if os.path.exists(save):
            print(file,"already exist.")
            continue
        print(" Processing ", file)
        outputs = process_features(os.path.join(path,file), similarity_method)
        print(f"\nSaving modified features ...")
        torch.save(outputs, save)