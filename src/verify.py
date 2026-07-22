# scripts/verify.py
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import setup_logging, seed_everything, get_device, ensure_dir
from src.tokenizer import AxiomTokenizer
from src.data import prepare_dataloader
import logging

logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging("INFO")
    cfg = load_config("configs/default.yaml")
    seed_everything(cfg.training.seed)
    device = get_device()
    
    ensure_dir(cfg.training.checkpoint_dir)
    ensure_dir("assets")
    
    logger.info("--- Starting Data Pipeline Test ---")
    
    tokenizer = AxiomTokenizer(vocab_size=cfg.model.vocab_size)
    
    # For testing, we just use a dummy text file so we don't wait for HF downloads
    dummy_file = "assets/dummy_corpus.txt"
    if not os.path.exists(dummy_file):
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 5000)
        tokenizer.train_from_files([dummy_file])
    else:
        tokenizer.load()

    # Test encoding
    test_ids = tokenizer.encode("Hello world, AxiomLLM is online.")
    logger.info(f"Encoded test string: {test_ids}")
    logger.info(f"Decoded back: {tokenizer.decode(test_ids)}")

    logger.info("--- Environment Verified. Ready for Model Architecture. ---")

if __name__ == "__main__":
    main()