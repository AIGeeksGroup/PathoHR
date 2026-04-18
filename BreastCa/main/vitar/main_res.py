from torch import nn, Tensor
from zeta import (
    MultiQueryAttention,
    FeedForward,
    patch_img,
)



class GridAttention(nn.Module):
    """
    GridAttention 模块在输入特征网格上应用注意力机制。

    args:
        dim (int): 输入特征的维度。
        heads (int, 可选): 注意力头的数量，默认为 8。
        dropout (float, 可选): 注意力模块中的 Dropout 比率，默认为 0.0。
        pre_norm (bool, 可选): 是否使用 Pre-LayerNorm 架构，默认为 False。
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, pre_norm: bool = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.pre_norm = pre_norm

        self.projection = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        GridAttention 模块的前向传播。

        args:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        return:
            Tensor: 应用注意力机制后的输出张量。
        """
        batch_size, seq_len, dim = x.shape
        x = self.norm(x)  # 如果启用 Pre-LayerNorm，则在此处归一化

        query = nn.AdaptiveAvgPool1d(dim)(x)  # 自适应平均池化生成查询向量
        key = self.projection(x)  # 生成键向量
        value = self.projection(x)  # 生成值向量

        attention_input = query + key + value  # 合并查询、键和值
        attention_output, _ = MultiQueryAttention(dim, self.heads)(attention_input)
        attention_output = self.attn_dropout(attention_output)  # 应用 Dropout
        return attention_output


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
    """
    AdaptiveTokenMerger 模块结合了网格注意力和前馈网络。

    args:
        dim (int): 输入特征的维度。
        heads (int): 注意力头的数量。
        dropout (float, 可选): Dropout 比率，默认为 0.1。
        norm (bool, 可选): 是否在模块之间添加 LayerNorm，默认为 True。
        residual (bool, 可选): 是否启用残差连接，默认为 True。
    """
    def __init__(self, dim: int, heads: int, dropout: float = 0.1, norm: bool = True, residual: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.norm = norm
        self.residual = residual

        self.attention = GridAttention(dim, heads, dropout=dropout, pre_norm=norm)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm_layer = nn.LayerNorm(dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        AdaptiveTokenMerger 模块的前向传播。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        返回:
            Tensor: 应用注意力和前馈网络后的输出张量。
        """
        residual = x
        attention_output = self.attention(x)  # 应用网格注意力
        attention_output = self.norm_layer(attention_output)  # 归一化

        feed_forward_output = self.feed_forward(attention_output)  # 应用前馈网络
        if self.residual:
            feed_forward_output = feed_forward_output + residual  # 残差连接

        return feed_forward_output


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

        # Norm
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    # def classifier(self, x: Tensor) -> Tensor:
    #     x = x.mean(dim = 1)

    #     x = self.to_latent(x)

    #     return self.linear_head(x)

    def forward(self, x) -> Tensor:
        # Embed the image -> (B, S, D)
        x = patch_img(x, self.patch_size)
        print(x.shape)

        b, s, d = x.shape

        # Norm
        # norm = nn.LayerNorm(d)

        # Adaptive token merger
        out = AdaptiveTokenMerger(d, self.heads, self.dropout)(x)
        print(out.shape)

        # Transformer Blocks
        transformed = TransformerBlocks(
            d,
            self.heads,
            self.dropout,
            self.depth,
        )(out)
        print(transformed.shape)

        # Start of classifier
        x = transformed.mean(dim=1)

        x = self.to_latent(x)

        return nn.Linear(d, self.num_classes)(x)
