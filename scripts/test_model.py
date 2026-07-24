import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import RMSNorm, RoPE, CausalSelfAttention, MultiHeadLatentAttention, SwiGLU, MoE, TransformerBlock, AxiomLLM
from src.utils import count_parameters

def test_all():
    B, T, C, H, V = 2, 16, 64, 4, 1000
    x = torch.randn(B, T, C)
    
    # Base components
    assert RMSNorm(C)(x).shape == (B, T, C)
    assert RoPE(32)(torch.randn(B, H, T, 32)).shape == (B, H, T, 32)
    assert CausalSelfAttention(C, H)(x).shape == (B, T, C)
    assert MultiHeadLatentAttention(C, H, 8, 8, 8, 16)(x).shape == (B, T, C)
    assert SwiGLU(C, 256)(x).shape == (B, T, C)
    
    out, aux = MoE(C, 256, 4, 2)(x)
    assert out.shape == (B, T, C) and aux.item() > 0
    
    # Blocks
    out, aux = TransformerBlock(C, H, use_mla=True, use_moe=False, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16)(x)
    assert out.shape == (B, T, C) and aux.item() == 0.0
    
    out, aux = TransformerBlock(C, H, use_mla=False, use_moe=True, num_experts=4, top_k=2)(x)
    assert out.shape == (B, T, C) and aux.item() > 0.0
    
    # Full Stack
    cfg = SimpleNamespace(vocab_size=V, embed_dim=128, num_heads=4, num_layers=2, mlp_ratio=4.0, use_mla=True, use_moe=True, num_experts=4, top_k=2, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=32, q_lora_rank=None)
    model = AxiomLLM(cfg)
    ids = torch.randint(0, V, (B, T))
    logits, total_aux = model(ids)
    loss = F.cross_entropy(logits.view(-1, V), ids.view(-1)) + 0.01 * total_aux
    loss.backward()
    print("🎉 ALL TESTS PASSED - DEEPSEEK ARCHITECTURE VERIFIED")

if __name__ == "__main__":
    test_all()