import yaml
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 4.0
    use_rope: bool = True
    use_mla: bool = True
    mla_rank: int = 32
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 3
    precision: str = "bf16"
    checkpoint_dir: str = "checkpoints/"
    seed: int = 42
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"

@dataclass
class AxiomConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(path: str = "configs/default.yaml") -> AxiomConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AxiomConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {}))
    )