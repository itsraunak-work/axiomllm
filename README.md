# AxiomLLM

A from-scratch, decoder-only transformer language model in PyTorch, built around a DeepSeek-V2/V3 style architecture: Multi-head Latent Attention (MLA) and a Mixture-of-Experts (MoE) feed-forward layer, in place of standard attention and a dense MLP. Written to be read line-by-line, not to hide behind a `Trainer` class.

## Architecture

- **RMSNorm** — pre-norm, learnable scale
- **RoPE** — rotary positional embeddings, applied per attention head
- **SwiGLU** — gated feed-forward block (`w2(SiLU(w1(x)) * w3(x))`), used inside each MoE expert instead of a plain GELU MLP
- **MLA (`MLA` class)** — instead of separate K/V projections per head, the input is down-projected into one shared low-rank latent (`latent_dim`, defaults to `embed_dim // 4`), then up-projected back into per-head K and V on the fly. This is what makes the KV cache small at inference time: you only need to store the latent, not full-size K/V. Note: this implementation applies RoPE directly to the reconstructed K (same as standard attention) rather than using DeepSeek-V2's "decoupled RoPE" trick, which splits Q/K into a rotated positional slice and an unrotated latent-derived slice. That decoupling is what lets DeepSeek fully avoid ever materializing K at inference; without it, K is still reconstructed via a cheap matmul each step, but the cache-size win is the same.
- **MoE (`MoE` class)** — 8 experts by default, top-2 routing. Each token's router logits go through softmax, top-2 experts are selected and renormalized, and only the tokens routed to a given expert are passed through it (via boolean masking) rather than running every expert on every token.
- **`DeepSeekBlock`** — `x = x + MLA(norm(x))`, then `x = x + MoE(norm(x))`
- **`AxiomLLM`** — token embedding → N `DeepSeekBlock`s → final RMSNorm → LM head

At the default config (vocab 50257, embed_dim 768, 12 layers, 8 experts/layer, top-2), this is roughly **550M total parameters** — heavier than a same-sized dense model, since each layer carries 8 full expert networks rather than one shared MLP. Forward pass confirmed working at this scale; backward pass needs more memory than a CPU-only environment comfortably provides — plan on a GPU for actual training runs.

## Training (`scripts/train.py`)

A real training loop, not just a forward-pass demo:
- AdamW with weight decay
- Mixed precision (fp16 with `GradScaler`, or bf16 without) via `torch.autocast`
- Gradient accumulation + gradient norm clipping
- Per-epoch checkpointing (`model_state_dict`, `optimizer_state_dict`, loss) to `checkpoint_dir`
- tqdm progress bar with running loss, structured logging via `setup_logging`
- Falls back to training a tiny BPE tokenizer on a generated dummy corpus if no tokenizer/dataset is available yet, so the script is runnable standalone before you point it at real data

## Data & tokenization

- `AxiomTokenizer` — custom BPE wrapper around Hugging Face `tokenizers`, trainable from raw text files, saved/loaded as a single JSON vocab file
- Streaming dataset loading (`datasets`, `streaming=True`) — no full local download needed before training starts
- `PackedDataset` — concatenates all tokenized text into one stream and slices fixed-length windows instead of padding, for better GPU utilization
- Default dataset: `roneneldan/TinyStories`

## Config system

- `ModelConfig` / `TrainingConfig` dataclasses (`src/config.py`), loaded from `configs/default.yaml`
- `use_mla` and `mla_rank` are currently vestigial — left over from an earlier version of the architecture. `AxiomLLM` always builds `DeepSeekBlock`s (MLA + MoE unconditionally), and MLA's latent dimension / MoE's expert count and top-k are hardcoded as constructor defaults rather than sourced from config yet

## Known issues

- **`lr: 3e-4` in `default.yaml` loads as a string, not a float** — PyYAML doesn't parse scientific notation without a decimal point (`3e-4`) as a float. This makes `torch.optim.AdamW(lr=cfg.training.lr, ...)` raise a `TypeError` the moment `train.py` runs. Fix: change to `3.0e-4` in the yaml.
- **`scripts/test_model.py` no longer matches `src/model.py`** — it still imports `CausalSelfAttention` and `TransformerBlock`, both removed in the MLA/MoE rewrite. The whole file fails at the import line, so no tests currently run at all.
- **No load-balancing loss in `MoE`** — the router has no auxiliary loss (or z-loss) encouraging even expert usage. With 8 experts and no balancing signal, router collapse (most tokens funneling into 1–2 experts, the rest going untrained) is a real risk once training actually starts.
- **No generation/inference script** — `AxiomLLM.forward` returns logits for the full sequence; there's no sampling loop (top-k/top-p/temperature) or KV-cache-based autoregressive decoding yet.

## Project structure

```
axiomllm/
├── configs/
│   └── default.yaml       # model + training hyperparameters
├── scripts/
│   ├── test_model.py      # unit tests (currently broken, see Known Issues)
│   ├── train.py            # full training loop: AMP, grad accumulation, checkpointing
│   └── verify.py            # environment/config sanity check
├── src/
│   ├── config.py            # dataclass config schema + YAML loader
│   ├── data.py                # streaming dataset + token packing
│   ├── model.py                 # RMSNorm, RoPE, SwiGLU, MLA, MoE, DeepSeekBlock, AxiomLLM
│   ├── tokenizer.py               # BPE tokenizer wrapper
│   ├── utils.py                     # logging, seeding, device detection, param counting
│   └── verify.py                     # data-pipeline-focused env check
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

Run the data pipeline check (trains a tokenizer on a dummy corpus if none exists):

```bash
python src/verify.py
```

Start training (fix the `lr` value in `configs/default.yaml` first — see Known Issues):

```bash
python scripts/train.py
```

## Design principles

- Interview-readable — non-obvious tensor reshapes get inline shape comments
- Config over constants, where the config is actually wired up
- Streaming-first data so a full dataset copy isn't required to start iterating
- Real architecture, not a toy — MLA and MoE are the actual mechanisms DeepSeek-V2/V3 use, simplified for readability but not faked

## License

MIT — see [LICENSE](LICENSE).