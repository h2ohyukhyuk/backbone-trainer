"""Hierarchical ViT with an inexpensive /4 stem and pooled-key/value attention.

Returns NCHW features in the same order/channels as APAG: (x4, e2, e3).
No class token: every output is supervised through average-pool classification.
"""
import torch
from torch import nn
from model.layers import DSConv


class AttentionProduct(nn.Module):
    """Explicit attention products, with a separate THOP counting rule."""
    def forward(self, q, k, v):
        weights = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        return weights.softmax(dim=-1) @ v


def count_attention(module, inputs, output):
    q, k, v = inputs
    batch, heads, queries, dim = q.shape
    module.total_ops += module.total_ops.new_tensor(
        [2 * batch * heads * queries * k.size(-2) * dim])


def profile_macs(model, input_shape=(1, 3, 256, 384)):
    """THOP MAC convention, including QK^T and attention-times-V products.

    As in THOP's CNN counts, this is an operation estimate, not latency;
    softmax/GELU and data movement are not included in multiply-add counts.
    Preserves model mode and supports its current device/dtype.
    """
    from thop import profile
    parameter = next(model.parameters())
    sample = torch.zeros(input_shape, device=parameter.device, dtype=parameter.dtype)
    modes = {module: module.training for module in model.modules()}
    try:
        macs, _ = profile(model, inputs=(sample,),
                          custom_ops={AttentionProduct: count_attention}, verbose=False)
    finally:
        for module, training in modes.items():
            module.training = training
    return macs, sum(p.numel() for p in model.parameters())


class PooledAttention(nn.Module):
    def __init__(self, dim, heads, pool_size):
        super().__init__()
        self.heads = heads
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.product = AttentionProduct()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, height, width):
        batch, tokens, dim = x.shape
        spatial = x.transpose(1, 2).reshape(batch, dim, height, width)
        # Sixteen averaged context tokens keep attention linear in image area.
        pooled = self.pool(spatial).flatten(2).transpose(1, 2)
        q = self.q(x).reshape(batch, tokens, self.heads, dim // self.heads).transpose(1, 2)
        kv = self.kv(pooled).reshape(batch, -1, 2, self.heads, dim // self.heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        out = self.product(q, k, v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, pool_size, hidden):
        super().__init__()
        # Depthwise positional encoding works with rectangular, variable inputs.
        self.position = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = PooledAttention(dim, heads, pool_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.position(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attention(self.norm1(tokens), height, width)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class APAGTF(nn.Module):
    def __init__(self, mlp_hidden=(117, 234, 475)):
        super().__init__()
        if len(mlp_hidden) != 3 or any(type(n) is not int or n <= 0 for n in mlp_hidden):
            raise ValueError('mlp_hidden must contain three positive integers')
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2), DSConv(16, 64),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            TransformerBlock(64, 2, 4, mlp_hidden[0]), nn.AvgPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 1), TransformerBlock(128, 4, 4, mlp_hidden[1]),
            nn.AvgPool2d(2, 2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 1), TransformerBlock(256, 8, 4, mlp_hidden[2]),
            nn.AvgPool2d(2, 2))
        self.layer2_extra = nn.Sequential(nn.Conv2d(64, 128, 1), nn.GELU())
        self.layer3_extra = nn.Sequential(nn.Conv2d(128, 128, 1), nn.GELU())

    def stem(self, x):
        return self.layer1(x)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4, self.layer2_extra(x2), self.layer3_extra(x3)
