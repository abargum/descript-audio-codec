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

class CausalMultiheadAttention(nn.Module):
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
                torch.ones(keylen, querylen, device=queries.device, dtype=torch.bool), diagonal=1
            )
            score.masked_fill_(causal_mask[None, None, :, :], -torch.inf)
        
        # Apply padding mask if provided
        if mask is not None:
            score.masked_fill_(~mask[:, None, :, :].to(torch.bool), -torch.inf)
            
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