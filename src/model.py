import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Any

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Cheaper than standard LayerNorm because it skips mean-centering.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_factor = torch.rsqrt(variance + self.eps)
        return x * norm_factor * self.weight


class RoPE(nn.Module):
    """
    Rotary Positional Embeddings.
    Applies 2D rotations to pairs of dimensions based on sequence position.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        
        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos
        
        rotated = torch.stack([x0_rot, x1_rot], dim=-1)
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """
    Standard Multi-Head Self Attention with Causal Masking.
    Used as a fallback or for architectural ablation studies.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q)
        k = self.rope(k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.out_proj(context)


class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2 style Multi-Head Latent Attention (MLA).
    Compresses K and V into a shared low-rank latent vector to shrink the KV cache.
    Uses Decoupled RoPE: splits Q/K into a compressible 'content' slice and an 
    uncompressed, shared 'positional' slice to bypass the non-commutativity of 
    rotation and low-rank projection matrices.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_lora_rank: int = 32,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        q_lora_rank: Optional[int] = None,
        max_seq_len: int = 2048,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scale = self.qk_head_dim ** -0.5

        # Query Path
        if q_lora_rank is not None:
            self.q_down_proj = nn.Linear(embed_dim, q_lora_rank, bias=False)
            self.q_down_norm = RMSNorm(q_lora_rank)
            self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_down_proj = None
            self.q_up_proj = nn.Linear(embed_dim, num_heads * self.qk_head_dim, bias=False)

        # KV Path (The MLA Core)
        self.kv_down_proj = nn.Linear(embed_dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_down_norm = RMSNorm(kv_lora_rank)
        self.kv_up_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        
        self.out_proj = nn.Linear(num_heads * v_head_dim, embed_dim, bias=False)
        self.rope = RoPE(qk_rope_head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # 1. Queries
        if self.q_down_proj is not None:
            q_latent = self.q_down_norm(self.q_down_proj(x))
            q = self.q_up_proj(q_latent)
        else:
            q = self.q_up_proj(x)
            
        q = q.view(batch, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = self.rope(q_rope)
        
        # 2. Keys & Values (Latent Compression)
        kv_down = self.kv_down_proj(x)
        c_kv, k_rope = torch.split(kv_down, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_down_norm(c_kv)  # THIS is what gets cached in VRAM
        
        kv = self.kv_up_proj(c_kv)
        kv = kv.view(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # 3. Decoupled RoPE Application
        k_rope = k_rope.unsqueeze(1)                    # Shared across heads
        k_rope = self.rope(k_rope)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)  # Broadcast without copying
        
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)
        
        # 4. Attention Math
        scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.v_head_dim)
        return self.out_proj(context)


class SwiGLU(nn.Module):
    """
    DeepSeek/LLaMA style Feed-Forward Network.
    Uses a gating mechanism (signal * gate) for better representation.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU is the Swish activation: x * sigmoid(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture of Experts.
    Routes tokens to Top-K experts for sparse, efficient compute.
    Returns the output and the auxiliary load-balancing loss.
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C) # [B*T, C]
        num_tokens = x_flat.size(0)
        
        # 1. Router calculates probabilities
        router_logits = self.router(x_flat) # [B*T, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize top-k weights so they sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 3. Calculate Auxiliary Load-Balancing Loss
        # f_i: fraction of tokens actually dispatched to expert i
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            tokens_per_expert += torch.bincount(topk_indices[:, k], minlength=self.num_experts)
        f_i = tokens_per_expert / (num_tokens * self.top_k)
        
        # P_i: mean routing probability for expert i
        P_i = routing_weights.mean(dim=0)
        
        # Balance loss = N * sum(f_i * P_i)
        aux_loss = (f_i * P_i).sum() * self.num_experts
        
        # 4. Route tokens and accumulate outputs
        out = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # Find which tokens were routed to Expert 'i'
            mask = (topk_indices == i).any(dim=-1) # Boolean mask [B*T]
            
            if mask.any():
                expert_input = x_flat[mask]
                expert_out = self.experts[i](expert_input)
                
                # Extract the specific routing weight for Expert 'i' for these tokens
                expert_idx_in_topk = (topk_indices[mask] == i).nonzero(as_tuple=True)[1]
                weight = topk_weights[mask, expert_idx_in_topk].unsqueeze(-1)
                
                out[mask] += expert_out * weight
                
        return out.view(B, T, C), aux_loss


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_mla: bool = False,
        use_moe: bool = False,       # NEW
        num_experts: int = 8,        # NEW
        top_k: int = 2,              # NEW
        kv_lora_rank: int = 32,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        q_lora_rank: int | None = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        
        if use_mla:
            self.attn = MultiHeadLatentAttention(
                embed_dim, num_heads, kv_lora_rank, qk_nope_head_dim, 
                qk_rope_head_dim, v_head_dim, q_lora_rank
            )
        else:
            self.attn = CausalSelfAttention(embed_dim, num_heads)
            
        self.norm2 = RMSNorm(embed_dim)
        self.use_moe = use_moe
        
        if use_moe:
            # MoE hidden dim calculation (approx 8/3 * dim for SwiGLU equivalence)
            moe_hidden_dim = int(embed_dim * 8 / 3)
            moe_hidden_dim = ((moe_hidden_dim + 255) // 256) * 256 # Tensor core alignment
            self.mlp = MoE(embed_dim, moe_hidden_dim, num_experts, top_k)
        else:
            hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = SwiGLU(embed_dim, hidden_dim) # Use SwiGLU even if not MoE

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))
        
        if self.use_moe:
            mlp_out, aux_loss = self.mlp(self.norm2(x))
            x = x + mlp_out
            return x, aux_loss
        else:
            x = x + self.mlp(self.norm2(x))
            return x, torch.tensor(0.0, device=x.device)


class AxiomLLM(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        
        # Read MoE settings from config, default to False if not present
        use_moe = getattr(cfg, 'use_moe', False)
        num_experts = getattr(cfg, 'num_experts', 8)
        top_k = getattr(cfg, 'top_k', 2)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                use_mla=cfg.use_mla,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k=top_k,
                kv_lora_rank=getattr(cfg, 'kv_lora_rank', 32),
                qk_nope_head_dim=getattr(cfg, 'qk_nope_head_dim', 32),
                qk_rope_head_dim=getattr(cfg, 'qk_rope_head_dim', 32),
                v_head_dim=getattr(cfg, 'v_head_dim', 64),
                q_lora_rank=getattr(cfg, 'q_lora_rank', None),
            ) for _ in range(cfg.num_layers)
        ])
        
        self.final_norm = RMSNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(input_ids)
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss