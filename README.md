# AxiomLLM

A from-scratch, decoder-only transformer language model built in PyTorch — written to be readable and testable line-by-line rather than to chase leaderboard numbers. No `Trainer` class, no black boxes: every non-trivial tensor operation is commented with its shape transformation.

## Why this exists

Most "build an LLM from scratch" projects stop at a notebook. This one is structured like something you'd actually maintain: dataclass-driven configs, a streaming data pipeline, and a test suite that checks correctness properties (not just shapes) — like verifying the causal mask actually blocks future tokens from leaking into past ones.

## What's implemented today

- **RMSNorm** — pre-norm, learnable scale
- **RoPE** (Rotary Positional Embeddings) — applied to Q/K per attention head
- **Causal multi-head self-attention** — manual QK^T scaling, triangular mask, softmax, weighted V sum
- **Pre-norm transformer block** — `x = x + attn(norm(x))`, `x = x + mlp(norm(x))`
- **MLP** — `Linear → GELU → Linear`, 4x hidden expansion

## Data & tokenization

- `AxiomTokenizer` — custom BPE tokenizer wrapping Hugging Face `tokenizers`, trainable from raw text files, saved/loaded as a single JSON vocab file
- **Streaming dataset loading** (`datasets`, `streaming=True`) — no need to download a dataset in full before training
- `PackedDataset` — concatenates all tokenized text into one stream and slices fixed-length windows instead of padding short sequences, for better GPU utilization
- Default dataset: `roneneldan/TinyStories`

## Config system

- `ModelConfig` / `TrainingConfig` dataclasses, loaded from `configs/default.yaml`
- Default model shape mirrors GPT-2 small: vocab 50257, embed_dim 768, 12 layers, 12 heads (~124M params)
- Every hyperparameter lives in config, not hardcoded in `src/`

## Testing

- `scripts/test_model.py` — unit tests for:
  - RMSNorm output shape + normalization property
  - RoPE output shape + position-0 identity (no rotation at position 0)
  - Attention output shape
  - **Causal mask leakage test**: corrupts the last token and asserts every earlier position's output is byte-identical — the test most people skip and interviewers actually ask about
  - Full forward + backward pass smoke test (embed → N blocks → LM head → cross-entropy → `.backward()`)
- `scripts/verify.py` / `src/verify.py` — environment sanity checks (device detection, effective batch size, tokenizer round-trip encode/decode)

## Roadmap — not yet built

Being upfront about the gap between config and code:

- **SwiGLU feed-forward** — MLP currently uses GELU; no SwiGLU implementation or config flag exists yet
- **MLA-style KV compression** — `use_mla` / `mla_rank` exist in `ModelConfig` but `CausalSelfAttention` doesn't consume them; attention is currently vanilla MHA with full-size K/V projections
- **Training loop** (`train.py`) — no script currently assembles embedding + blocks + LM head into a trainable model outside of the test suite
- **Checkpointing** — save/load is referenced (`checkpoint_dir` in config) but not implemented
- **Generation / sampling** — no inference script (top-k, top-p, temperature)
- **KV-cache** — needed for non-quadratic autoregressive inference once generation exists

## Project structure

```
axiomllm/
├── configs/
│   └── default.yaml       # model + training hyperparameters
├── scripts/
│   ├── test_model.py      # unit tests for model components
│   └── verify.py          # environment/config sanity check
├── src/
│   ├── config.py           # dataclass config schema + YAML loader
│   ├── data.py              # streaming dataset + token packing
│   ├── model.py              # RMSNorm, RoPE, attention, transformer block
│   ├── tokenizer.py           # BPE tokenizer wrapper
│   ├── utils.py                 # seeding, device detection, param counting
│   └── verify.py                 # data-pipeline-focused env check
├── requirements.txt
└── LICENSE (MIT)
```

## Installation

```bash
git clone https://github.com/itsraunak-work/axiomllm.git
cd axiomllm
pip install -r requirements.txt
```

## Usage

Check your environment and config load correctly:

```bash
python scripts/verify.py
```

Run the model unit test suite:

```bash
python scripts/test_model.py
```

## Design principles

- **Interview-readable** — every attention/RoPE tensor reshape has an inline shape comment
- **Config over constants** — no magic numbers buried in `src/`
- **Streaming-first data** — don't require a full local dataset copy to start iterating
- **Correctness tests before scale** — the causal-mask leakage test exists before there's even a training loop

## License

MIT — see [LICENSE](LICENSE).