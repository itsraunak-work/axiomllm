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




class SwiGLU(nn.Module):
    """
    DeepSeek/LLaMA style Feed-Forward Network.
    Uses a gating mechanism for better gradient flow and representation.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # We use 3 matrices. To keep FLOPs similar to standard MLP, 
        # hidden_dim is usually set to (8/3) * dim, rounded to nearest multiple of 256.
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU is the Swish activation: x * sigmoid(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLA(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2 style).
    Uses low-rank compression for the KV Cache to save massive amounts of VRAM.
    """
    def __init__(self, embed_dim: int, num_heads: int, latent_dim: int = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # If latent_dim not specified, default to 1/4 of embed_dim (DeepSeek ratio)
        self.latent_dim = latent_dim or (embed_dim // 4)

        # Query Projections (Standard)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # KV Joint Compression (The MLA Magic)
        # 1. Down-project to tiny latent space
        self.kv_down_proj = nn.Linear(embed_dim, self.latent_dim, bias=False)
        # 2. Up-project back to full head dimensions on the fly
        self.k_up_proj = nn.Linear(self.latent_dim, embed_dim, bias=False)
        self.v_up_proj = nn.Linear(self.latent_dim, embed_dim, bias=False)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # 1. Standard Query generation
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. MLA: Compress to Latent Space (This is what goes into the VRAM Cache)
        latent_kv = self.kv_down_proj(x) # Shape: [B, T, latent_dim]
        
        # 3. MLA: Reconstruct K and V on the fly in GPU Registers
        k = self.k_up_proj(latent_kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up_proj(latent_kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)
        
        # 5. Attention Math (Identical to standard, but K/V came from latent space)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(context)


class MoE(nn.Module):
    """
    Mixture of Experts.
    Routes tokens to Top-K experts for sparse, efficient compute.
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The API Gateway (Router)
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # The Microservices (Experts)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C) # [B*T, C]
        
        # 1. Router calculates probabilities for each expert
        router_logits = self.router(x_flat) # [B*T, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K experts for each token
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1) # [B*T, K]
        
        # 3. Normalize the top-k weights so they sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 4. Route tokens and accumulate outputs
        out = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # Find which tokens were routed to Expert 'i'
            mask = (topk_indices == i).any(dim=-1) # Boolean mask [B*T]
            
            if mask.any():
                # Process only the routed tokens through this expert
                expert_input = x_flat[mask]
                expert_out = self.experts[i](expert_input)
                
                # Find the specific routing weight for Expert 'i' for these tokens
                # (We need to extract the weight from the topk_weights tensor)
                expert_idx_in_topk = (topk_indices[mask] == i).nonzero(as_tuple=True)[1]
                weight = topk_weights[mask, expert_idx_in_topk].unsqueeze(-1)
                
                # Accumulate the weighted expert output
                out[mask] += expert_out * weight
                
        return out.view(B, T, C)


class DeepSeekBlock(nn.Module):
    """
    AxiomLLM Transformer Block using MLA and MoE.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MLA(embed_dim, num_heads)
        
        self.norm2 = RMSNorm(embed_dim)
        
        # MoE hidden dim calculation (approx 8/3 * dim for SwiGLU equivalence)
        moe_hidden_dim = int(embed_dim * 8 / 3)
        # Round to nearest multiple of 256 for GPU tensor core alignment
        moe_hidden_dim = ((moe_hidden_dim + 255) // 256) * 256 
        
        self.moe = MoE(embed_dim, moe_hidden_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class AxiomLLM(nn.Module):
    """
    The complete autoregressive language model.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, num_experts: int = 8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Stack DeepSeek Blocks
        self.layers = nn.ModuleList([
            DeepSeekBlock(embed_dim, num_heads, num_experts) for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits