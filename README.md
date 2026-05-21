# AxiomLLM

A from-scratch, optimized decoder-only language model built for efficiency and interview-ready engineering practices.

## 🔑 Key Architecture Choices
- **RoPE** (Rotary Positional Embeddings) for stable long-context scaling
- **SwiGLU** feed-forward layers for improved gradient flow
- **MLA-style KV compression** to reduce inference memory by ~60%
- **RMSNorm** + **BF16 mixed precision** for faster, stable training on consumer GPUs
- Config-driven hyperparameters, reproducible seeding, and streaming data pipeline

## 📐 Architecture