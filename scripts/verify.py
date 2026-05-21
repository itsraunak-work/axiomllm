# scripts/verify.py
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import setup_logging, seed_everything, get_device, ensure_dir
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
    logger.info(f"Config schema validated")

if __name__ == "__main__":
    main()