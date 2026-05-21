import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import setup_logging, seed_everything, get_device, ensure_dir, count_parameters
import logging

logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging("INFO")
    cfg = load_config("configs/default.yaml")
    seed_everything(cfg.training.seed)
    device = get_device()
    
    ensure_dir(cfg.training.checkpoint_dir)
    ensure_dir("assets")
    
    logger.info(f"AxiomLLM environment verified")
    logger.info(f"Device: {device}")
    logger.info(f"Precision: {cfg.training.precision}")
    logger.info(f"Effective batch size: {cfg.training.batch_size * cfg.training.gradient_accumulation_steps}")
    logger.info(f"Model params (total/trainable): {count_parameters}")
    logger.info(f"Config schema validated")

if __name__ == "__main__":
    main()