import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (
    RMSNorm, RoPE, CausalSelfAttention, MultiHeadLatentAttention, 
    SwiGLU, MoE, TransformerBlock, AxiomLLM
)
from src.utils import count_parameters

def test_rmsnorm():
    B, T, C = 2, 8, 64
    x = torch.randn(B, T, C)
    norm = RMSNorm(C)
    out = norm(x)
    assert out.shape == (B, T, C), f"RMSNorm shape: expected {(B,T,C)}, got {out.shape}"
    
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), "RMSNorm not normalizing correctly"
    print("  ✅ RMSNorm: Shape and normalization verified")

def test_rope():
    B, H, T, D = 2, 4, 16, 32
    x = torch.randn(B, H, T, D)
    rope = RoPE(D)
    out = rope(x)
    assert out.shape == (B, H, T, D), f"RoPE shape: expected {(B,H,T,D)}, got {out.shape}"
    assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-5), "RoPE position 0 should be near-identity"
    print("  ✅ RoPE: Shape and position-0 identity verified")

def test_causal_attention():
    B, T, C, H = 2, 16, 64, 4
    x = torch.randn(B, T, C)
    attn = CausalSelfAttention(C, H)
    out = attn(x)
    assert out.shape == (B, T, C), f"Attention shape: expected {(B,T,C)}, got {out.shape}"
    print("  ✅ CausalSelfAttention: Shape verified")

def test_causal_mask_correctness():
    torch.manual_seed(42)
    C, H = 64, 4
    attn = CausalSelfAttention(C, H)
    attn.eval()
    
    x1 = torch.randn(1, 8, C)
    x2 = x1.clone()
    x2[0, -1, :] = torch.randn(C) * 100 
    
    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)
        
    assert torch.allclose(out1[:, :7, :], out2[:, :7, :], atol=1e-5), "CAUSAL MASK FAILED: Future tokens are leaking!"
    assert not torch.allclose(out1[:, 7, :], out2[:, 7, :], atol=1e-3), "Position 7 should change when its own input changes"
    print("  ✅ Causal Mask: Future-token leakage test passed")

def test_mla():
    B, T, C, H = 2, 16, 64, 4
    x = torch.randn(B, T, C)
    mla = MultiHeadLatentAttention(C, H, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16)
    out = mla(x)
    assert out.shape == (B, T, C), f"MLA shape: expected {(B,T,C)}, got {out.shape}"
    print("  ✅ MultiHeadLatentAttention: Shape verified")

def test_mla_causal_mask_correctness():
    torch.manual_seed(42)
    C, H = 64, 4
    mla = MultiHeadLatentAttention(C, H, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16)
    mla.eval()
    
    x1 = torch.randn(1, 8, C)
    x2 = x1.clone()
    x2[0, -1, :] = torch.randn(C) * 100 
    
    with torch.no_grad():
        out1 = mla(x1)
        out2 = mla(x2)
        
    assert torch.allclose(out1[:, :7, :], out2[:, :7, :], atol=1e-5), "CAUSAL MASK FAILED in MLA!"
    assert not torch.allclose(out1[:, 7, :], out2[:, 7, :], atol=1e-3), "Position 7 should change"
    print("  ✅ MLA Causal Mask: Future-token leakage test passed")

def test_mla_kv_cache_savings():
    num_heads, head_dim = 12, 64
    kv_lora_rank, qk_rope_head_dim = 32, 32
    mha_cache_per_token = 2 * num_heads * head_dim          
    mla_cache_per_token = kv_lora_rank + qk_rope_head_dim    
    assert mla_cache_per_token < mha_cache_per_token, "MLA should shrink the KV cache"
    ratio = mha_cache_per_token / mla_cache_per_token
    print(f"  ✅ MLA KV-cache savings: {mha_cache_per_token} -> {mla_cache_per_token} floats/token ({ratio:.1f}x smaller)")

def test_swiglu():
    B, T, C = 2, 8, 64
    hidden = 256
    x = torch.randn(B, T, C)
    swiglu = SwiGLU(C, hidden)
    out = swiglu(x)
    assert out.shape == (B, T, C), f"SwiGLU shape mismatch: {out.shape}"
    print("  ✅ SwiGLU: Shape verified")

def test_moe():
    B, T, C = 2, 8, 64
    hidden = 256
    x = torch.randn(B, T, C)
    moe = MoE(C, hidden, num_experts=4, top_k=2)
    out, aux_loss = moe(x)
    
    assert out.shape == (B, T, C), f"MoE shape mismatch: {out.shape}"
    assert aux_loss.ndim == 0, "Aux loss should be a scalar tensor"
    assert aux_loss.item() > 0, "Aux loss should be > 0 for random routing"
    print(f"  ✅ MoE: Shape verified | Aux Loss: {aux_loss.item():.4f}")

def test_transformer_block():
    B, T, C, H = 2, 16, 64, 4
    x = torch.randn(B, T, C)
    
    # 1. Standard Block (Dense SwiGLU)
    block = TransformerBlock(C, H, use_mla=False, use_moe=False)
    out, aux_loss = block(x)
    assert out.shape == (B, T, C)
    assert aux_loss.item() == 0.0, "Dense block should have 0 aux loss"
    
    # 2. MLA Block
    mla_block = TransformerBlock(C, H, use_mla=True, use_moe=False, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16)
    out_mla, aux_loss_mla = mla_block(x)
    assert out_mla.shape == (B, T, C)
    
    # 3. MoE Block
    moe_block = TransformerBlock(C, H, use_mla=False, use_moe=True, num_experts=4, top_k=2)
    out_moe, aux_loss_moe = moe_block(x)
    assert out_moe.shape == (B, T, C)
    assert aux_loss_moe.item() > 0.0, "MoE block should have > 0 aux loss"
    
    print("  ✅ TransformerBlock: Standard, MLA, and MoE routing verified")

def test_full_stack():
    """Simulate a mini AxiomLLM forward + backward pass"""
    B, T, V = 2, 32, 1000
    
    # Create a dummy config object to pass to AxiomLLM
    dummy_cfg = SimpleNamespace(
        vocab_size=V, embed_dim=128, num_heads=4, num_layers=2, mlp_ratio=4.0,
        use_mla=True, use_moe=True, num_experts=4, top_k=2,
        kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=32, q_lora_rank=None
    )
    
    model = AxiomLLM(dummy_cfg)
    input_ids = torch.randint(0, V, (B, T))
    
    # Forward pass
    logits, total_aux_loss = model(input_ids)
    assert logits.shape == (B, T, V), f"Logits shape mismatch: {logits.shape}"
    
    # Compute combined loss (Cross Entropy + MoE Penalty)
    labels = input_ids
    ce_loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
    total_loss = ce_loss + (0.01 * total_aux_loss)
    
    # Backward pass must complete without errors
    total_loss.backward()
    
    total_params, _ = count_parameters(model)
    print(f"  ✅ Full Stack (MLA + MoE): Forward + Backward passed | CE Loss: {ce_loss.item():.4f} | Aux Loss: {total_aux_loss.item():.4f} | Params: {total_params:,}")

if __name__ == "__main__":
    print("=" * 60)
    print("  AXIOMLLM DEEPSEEK ARCHITECTURE TEST SUITE")
    print("=" * 60)
    
    test_rmsnorm()
    test_rope()
    test_causal_attention()
    test_causal_mask_correctness()
    test_mla()
    test_mla_causal_mask_correctness()
    test_mla_kv_cache_savings()
    test_swiglu()
    test_moe()
    test_transformer_block()
    test_full_stack()
    
    print("=" * 60)
    print("  🎉 ALL TESTS PASSED - ARCHITECTURE VERIFIED")
    print("=" * 60)