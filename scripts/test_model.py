import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import RMSNorm, RoPE, CausalSelfAttention, TransformerBlock
from src.utils import count_parameters


def test_rmsnorm():
    B, T, C = 2, 8, 64
    x = torch.randn(B, T, C)
    norm = RMSNorm(C)
    out = norm(x)
    assert out.shape == (B, T, C), f"RMSNorm shape: expected {(B,T,C)}, got {out.shape}"
    
    # Verify normalization property: RMS of output should be approximately 1 (before weight)
    # With weight=1, the RMS of each vector should be close to 1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), "RMSNorm not normalizing correctly"
    print("  ✅ RMSNorm: Shape and normalization verified")


def test_rope():
    B, H, T, D = 2, 4, 16, 32
    x = torch.randn(B, H, T, D)
    rope = RoPE(D)
    out = rope(x)
    assert out.shape == (B, H, T, D), f"RoPE shape: expected {(B,H,T,D)}, got {out.shape}"
    
    # Verify position 0 is approximately identity (cos(0)=1, sin(0)=0)
    # At position 0, rotation angle is 0, so output should equal input
    assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-5), \
        "RoPE position 0 should be near-identity"
    print("  ✅ RoPE: Shape and position-0 identity verified")


def test_causal_attention():
    B, T, C, H = 2, 16, 64, 4
    x = torch.randn(B, T, C)
    attn = CausalSelfAttention(C, H)
    out = attn(x)
    assert out.shape == (B, T, C), f"Attention shape: expected {(B,T,C)}, got {out.shape}"
    print("  ✅ CausalSelfAttention: Shape verified")


def test_causal_mask_correctness():
    """
    The most critical test: Verify that future tokens cannot influence past tokens.
    If we change token at position T, all positions before T must have identical output.
    """
    torch.manual_seed(42)
    C, H = 64, 4
    attn = CausalSelfAttention(C, H)
    attn.eval()  # Disable dropout if any
    
    # Create two inputs that differ only at the last position
    x1 = torch.randn(1, 8, C)
    x2 = x1.clone()
    x2[0, -1, :] = torch.randn(C) * 100  # Corrupt last token heavily
    
    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)
    
    # Positions 0 through 6 must be identical (they cannot see position 7)
    assert torch.allclose(out1[:, :7, :], out2[:, :7, :], atol=1e-5), \
        "CAUSAL MASK FAILED: Future tokens are leaking into past positions!"
    
    # Position 7 SHOULD be different (it sees itself)
    assert not torch.allclose(out1[:, 7, :], out2[:, 7, :], atol=1e-3), \
        "Position 7 should change when its own input changes"
    
    print("  ✅ Causal Mask: Future-token leakage test passed")


def test_transformer_block():
    B, T, C, H = 2, 16, 64, 4
    x = torch.randn(B, T, C)
    block = TransformerBlock(C, H)
    out = block(x)
    assert out.shape == (B, T, C), f"Block shape: expected {(B,T,C)}, got {out.shape}"
    
    total, trainable = count_parameters(block)
    print(f"  ✅ TransformerBlock: Shape verified | Params: {total:,} total, {trainable:,} trainable")


def test_full_stack():
    """Simulate a mini AxiomLLM forward pass"""
    B, T, V, C, H, L = 2, 32, 1000, 128, 4, 3
    
    embed = torch.nn.Embedding(V, C)
    layers = torch.nn.ModuleList([TransformerBlock(C, H) for _ in range(L)])
    head = torch.nn.Linear(C, V, bias=False)
    
    input_ids = torch.randint(0, V, (B, T))
    
    # Forward pass
    x = embed(input_ids)
    for layer in layers:
        x = layer(x)
    logits = head(x)
    
    assert logits.shape == (B, T, V), f"Logits shape: expected {(B,T,V)}, got {logits.shape}"
    
    # Compute loss (cross-entropy with causal shift)
    labels = input_ids
    loss = torch.nn.functional.cross_entropy(logits.view(-1, V), labels.view(-1))
    
    # Backward pass must complete without errors
    loss.backward()
    
    total, _ = count_parameters(torch.nn.ModuleList([embed, layers, head]))
    print(f"  ✅ Full Stack: Forward + Backward passed | Loss: {loss.item():.4f} | Total params: {total:,}")


if __name__ == "__main__":
    print("=" * 50)
    print("  AXIOMLLM MODEL TEST SUITE")
    print("=" * 50)
    
    test_rmsnorm()
    test_rope()
    test_causal_attention()
    test_causal_mask_correctness()
    test_transformer_block()
    test_full_stack()
    
    print("=" * 50)
    print("  🎉 ALL TESTS PASSED")
    print("=" * 50)