import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, dim]
        # Step 1: Square every element, then take mean along last dim
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Step 2: Add epsilon, take reciprocal square root
        norm_factor = torch.rsqrt(variance + self.eps)
        # Step 3: Normalize and apply learnable scale
        return x * norm_factor * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        # dim here is head_dim (must be even)
        assert dim % 2 == 0, "RoPE requires even head_dim"
        
        # Calculate inverse frequencies: theta_i = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = x.shape
        
        # Step 1: Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Step 2: Compute angles: [seq_len, head_dim/2]
        # Each position gets multiplied by each frequency
        angles = torch.outer(positions, self.inv_freq)
        
        # Step 3: Compute cos and sin, expand for broadcasting
        # Shape: [1, 1, seq_len, head_dim/2]
        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        
        # Step 4: Split x into even and odd dimensions
        # x0 = dimensions [0, 2, 4, ...], x1 = dimensions [1, 3, 5, ...]
        x0 = x[..., 0::2]  # Shape: [batch, num_heads, seq_len, head_dim/2]
        x1 = x[..., 1::2]  # Shape: [batch, num_heads, seq_len, head_dim/2]
        
        # Step 5: Apply 2D rotation
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos
        
        # Step 6: Interleave back together
        # Stack along last dim: [batch, num_heads, seq_len, head_dim/2, 2]
        rotated = torch.stack([x0_rot, x1_rot], dim=-1)
        # Flatten last two dims: [batch, num_heads, seq_len, head_dim]
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(head_dim)
        
        # Separate projections for clarity (fused QKV is a production optimization)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape
        
        # Step 1: Project input into Q, K, V
        # Each has shape: [batch, seq_len, embed_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Step 2: Reshape for multi-head attention
        # [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        
        # Step 3: Transpose to [batch, num_heads, seq_len, head_dim]
        # This puts heads in the batch dimension for parallel matmul
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Step 4: Apply RoPE to Q and K (injects positional information)
        q = self.rope(q)
        k = self.rope(k)
        
        # Step 5: Compute attention scores
        # Q @ K^T -> [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Step 6: Apply causal mask (prevent looking at future tokens)
        # Create lower-triangular boolean mask: [seq_len, seq_len]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        # Invert: True where we want to mask (upper triangle)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Step 7: Softmax to get attention weights (probabilities)
        # Shape: [batch, num_heads, seq_len, seq_len]
        weights = F.softmax(scores, dim=-1)
        
        # Step 8: Weighted sum of Values
        # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        context = torch.matmul(weights, v)
        
        # Step 9: Transpose back and reshape to original embedding dim
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2).contiguous()
        # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, embed_dim]
        context = context.view(batch, seq_len, embed_dim)
        
        # Step 10: Final linear projection (mix information across heads)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # Sub-layer 1: Attention with Pre-Norm
        self.norm1 = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        
        # Sub-layer 2: MLP with Pre-Norm
        self.norm2 = RMSNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm architecture: Normalize BEFORE the sub-layer
        # Residual connection: Add original input to sub-layer output
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x