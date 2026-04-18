from torch import nn, Tensor
import torch
from zeta import (
    MultiQueryAttention,
    FeedForward,
    patch_img,
)
import os

class GridAttention(nn.Module):
    """
    GridAttention module applies attention mechanism on a grid of input features.

    Args:
        dim (int): The dimension of the input features.
        heads (int, optional): The number of attention heads. Defaults to 8.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # Projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the GridAttention module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: The output tensor after applying attention mechanism.
        """
        b, s, d = x.shape

        k = self.proj(x)
        v = self.proj(x)

        # Average pool
        q = nn.AdaptiveAvgPool1d(d)(x)
        print(x.shape)

        out, _, _ = MultiQueryAttention(d, self.heads)(q + k + v)
        return out


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


class AdaptiveTokenMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout

        # grid attention
        self.attn = GridAttention(dim, heads)

        # Ffn
        self.ffn = FeedForward(dim, dim, 4, swish=True)

    def forward(self, x: Tensor) -> Tensor:
        print('x:',x.shape)
        # Grid Attention
        grid = self.attn(x)

        # Ffn
        ffn = self.ffn(grid)
        print('ffn:',ffn.shape)

        return ffn


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

class FeatureVitar(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        heads: int = 8,
        dropout: float = 0.1,
        depth: int = 9,
        num_classes: int = 2,
        max_tokens: int = 512
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.num_classes = num_classes
        self.max_tokens = max_tokens

        self.token_merger = AdaptiveTokenMerger(dim, heads, dropout)
        self.transformer_blocks = TransformerBlocks(
            dim,
            heads,
            dropout,
            depth,
        )
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        print(f"Input features shape: {features.shape}")
        
        # Apply token merger
        merged = self.token_merger(features)
        print(f"After token merger shape: {merged.shape}")
        
        # Transform
        transformed = self.transformer_blocks(merged)
        print(f"After transformer blocks shape: {transformed.shape}")
        
        # Pool and classify
        pooled = transformed.mean(dim=1)
        print(f"After pooling shape: {pooled.shape}")
        
        latent = self.to_latent(pooled)
        output = self.linear_head(latent)
        print(f"Final output shape: {output.shape}")
        
        return output

# Direct usage with single feature file
def process_features(feature_path: str, sequence_length: int = 512):
    # Load features
    print("Loading features...")
    features = torch.load(feature_path)
    print(f"Loaded features shape: {features.shape}")
    
    # Reshape features into sequences
    total_patches = features.size(0)
    dim = features.size(1)
    
    # Calculate number of complete sequences
    num_sequences = total_patches // sequence_length
    if num_sequences == 0:
        sequence_length = total_patches  # If total patches less than sequence_length
        num_sequences = 1
    
    print(f"Reshaping into {num_sequences} sequences of length {sequence_length}")
    features = features[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, dim)
    print(f"Reshaped features shape: {features.shape}")
    
    # Initialize model
    model = FeatureVitar(
        dim=dim,
        heads=8,
        depth=9,
        num_classes=2  # Adjust as needed
    )
    
    # Process features
    print("\nProcessing features through model...")
    with torch.no_grad():  # Inference mode
        outputs = model(features)
    
    print("\nProcessing complete!")
    return outputs

# Example usage
if __name__ == "__main__":
    # Process your feature file
    path = r"" # feature path
    for file in os.listdir(path):
        print(" Processing ", file)
        outputs = process_features(os.path.join(path,file))
        print(' Done for ', file)
    