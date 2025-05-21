import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type
import collections.abc
from itertools import repeat
import math

# Code from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            use_fused_attn: bool = True,
            causal: bool = True,  # Added causal flag, default to True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn
        self.causal = causal  # Store causal flag
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            # Use the is_causal parameter in scaled_dot_product_attention
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=self.causal,  # Pass the causal flag
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            # Apply causal mask if needed
            if self.causal:
                # Create a causal mask (lower triangular including the diagonal)
                mask = torch.ones((N, N), device=x.device, dtype=torch.bool).triu_(diagonal=1)
                # Fill upper triangular with -infinity to create causal attention
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#Dit Block from https://github.com/facebookresearch/DiT/blob/main/models.py

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Modified to support causal attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, causal=True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, causal=causal, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x)
        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x.transpose(2, 1)


class CausalMultiheadAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim=None,
        value_dim=None,
        embed_dim=None,
        num_heads=8,
        dropout=0.0,
        bias=True
    ):
        super().__init__()
        # Handle default values for dimensions
        key_dim = key_dim if key_dim is not None else query_dim
        value_dim = value_dim if value_dim is not None else key_dim
        embed_dim = embed_dim if embed_dim is not None else query_dim
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Save parameters
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(query_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(key_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(value_dim, embed_dim, bias=bias)
        
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x_q, x_k=None, x_v=None):

        """
        x_q: query tensor of shape (B, T_q, query_dim)
        x_k: key tensor of shape (B, T_kv, key_dim), defaults to x_q if None
        x_v: value tensor of shape (B, T_kv, value_dim), defaults to x_k if None
        
        Returns: output tensor of shape (B, T_q, embed_dim)
        """
        # Default to self-attention if no key/value provided
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_k
            
        B, T_q, _ = x_q.size()  # batch size, query sequence length
        _, T_kv, _ = x_k.size()  # key/value sequence length
        
        q = self.q_proj(x_q)  # (B, T_q, embed_dim)
        k = self.k_proj(x_k)  # (B, T_kv, embed_dim)
        v = self.v_proj(x_v)  # (B, T_kv, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T_q, hd)
        k = k.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T_kv, hd)
        v = v.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T_kv, hd)
        
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        y = self.resid_dropout(self.c_proj(y))
        
        return y.transpose(2, 1)

class CausalMultiheadAttention2(nn.Module):
    """Causal Multi-head scaled dot-product attention.
    """
    def __init__(self,
                 keys: int,
                 values: int,
                 queries: int,
                 out_channels: int,
                 hiddens: int,
                 heads: int,
                 dropout: float = 0.0):
        """Initializer.
        Args:
            keys, values, queries: size of the input channels.
            out_channels: size of the output channels.
            hiddens: size of the hidden channels.
            heads: the number of the attention heads.
            dropout: dropout probability.
        """
        super().__init__()
        assert hiddens % heads == 0, \
            f'size of hiddens channels(={hiddens}) should be factorized by heads(={heads})'
        self.channels, self.heads = hiddens // heads, heads
        self.proj_key = nn.Conv1d(keys, hiddens, 1)
        self.proj_value = nn.Conv1d(values, hiddens, 1)
        self.proj_query = nn.Conv1d(queries, hiddens, 1)
        self.proj_out = nn.Conv1d(hiddens, out_channels, 1)
        
        # Adding dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self,
                keys: torch.Tensor,
                values: torch.Tensor,
                queries: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform the inputs with causal attention.
        Args:
            keys: [torch.float32; [B, keys, S]], attention key.
            values: [torch.float32; [B, values, S]], attention value.
            queries: [torch.float32; [B, queries, T]], attention query.
            mask: [torch.float32; [B, S, T]], attention mask, 0 for paddings.
        Returns:
            [torch.float32; [B, out_channels, T]], transformed outputs.
        """
        # B, T
        bsize, _, querylen = queries.shape
        # S
        keylen = keys.shape[-1]
        assert keylen == values.shape[-1], 'lengths of key and value are not matched'
        
        # [B, H, hiddens // H, S]
        keys = self.proj_key(keys).view(bsize, self.heads, -1, keylen)
        values = self.proj_value(values).view(bsize, self.heads, -1, keylen)
        
        # [B, H, hiddens // H, T]
        queries = self.proj_query(queries).view(bsize, self.heads, -1, querylen)
        
        # [B, H, S, T]
        score = torch.matmul(keys.transpose(2, 3), queries) * (self.channels ** -0.5)
        
        # Apply causal mask - ensure each position can only attend to previous positions
        # Create a causal mask that prevents attending to future tokens
        if keylen == querylen:  # Self-attention case
            causal_mask = torch.triu(
                torch.ones(keylen, querylen, device=queries.device), diagonal=1
            ).bool()
            score.masked_fill_(causal_mask[None, None, :, :], -np.inf)
        
        # Apply padding mask if provided
        if mask is not None:
            score.masked_fill_(~mask[:, None, :, :].to(torch.bool), -np.inf)
            
        # [B, H, S, T]
        weights = torch.softmax(score, dim=2)
        weights = self.attn_dropout(weights)
        
        # [B, hiddens, T]
        out = torch.matmul(values, weights).view(bsize, -1, querylen)
        
        # Output projection with dropout
        out = self.resid_dropout(self.proj_out(out))
        
        if mask is not None:
            out = out * mask[:, :1]
            
        return out