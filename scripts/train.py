# scripts/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import setup_logging, seed_everything, get_device, ensure_dir
from src.tokenizer import AxiomTokenizer
from src.data import prepare_dataloader
from src.model import AxiomLLM

logger = logging.getLogger(__name__)

def train():
    # 1. Setup & Config
    setup_logging("INFO")
    cfg = load_config("configs/default.yaml")
    seed_everything(cfg.training.seed)
    device = get_device()
    
    # Ensure checkpoint directory exists (Will map to Google Drive in Colab)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ensure_dir(ckpt_dir)

    # 2. Initialize Tokenizer & Data
    logger.info("Loading Tokenizer and Data Pipeline...")
    tokenizer = AxiomTokenizer(vocab_size=cfg.model.vocab_size)
    
    # Fallback to dummy data if HF dataset fails or for quick testing
    dummy_file = "assets/dummy_corpus.txt"
    if not os.path.exists("assets/axiom_tokenizer.json"):
        if not os.path.exists(dummy_file):
            with open(dummy_file, "w", encoding="utf-8") as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 5000)
        tokenizer.train_from_files([dummy_file])
    else:
        tokenizer.load()

    dataloader = prepare_dataloader(
        dataset_name=cfg.training.dataset_name,
        split=cfg.training.dataset_split,
        tokenizer=tokenizer,
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        max_samples=5000 # Limit for quick Colab validation
    )

    # 3. Initialize Model
    logger.info(f"Initializing AxiomLLM on {device}...")
    #model = AxiomLLM(
    #    vocab_size=cfg.model.vocab_size,
    #    embed_dim=cfg.model.embed_dim,
    #    num_heads=cfg.model.num_heads,
    #    num_layers=cfg.model.num_layers
    #).to(device)
    model = AxiomLLM(cfg.model).to(device)

    # 4. Optimizer & Scaler
    # AdamW decouples weight decay from gradient updates, crucial for Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    
    # We use FP16 + GradScaler for compatibility with Colab's Tesla T4 GPUs
    # (If you upgrade to an A100 in Colab Pro, you can switch to BF16 and drop the scaler)
    use_amp = (device.type == 'cuda') and (cfg.training.precision in ["fp16", "bf16"])
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and cfg.training.precision == "fp16"))
    amp_dtype = torch.float16 if cfg.training.precision == "fp16" else torch.bfloat16

    # 5. Training Loop
    logger.info("Starting Training Loop...")
    model.train()
    global_step = 0
    
    for epoch in range(cfg.training.max_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad() # Clear gradients at start of epoch
        
        for step, (input_ids, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward Pass with Mixed Precision
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits, aux_loss = model(input_ids)
                
                # Calculate standard Cross-Entropy Loss
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Add Auxiliary Load-Balancing Loss (scaled by alpha=0.01)
                loss = ce_loss + (0.01 * aux_loss)
                
                loss = loss / cfg.training.gradient_accumulation_steps
            
            # Backward Pass
            if use_amp and cfg.training.precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            epoch_loss += loss.item() * cfg.training.gradient_accumulation_steps
            
            # Gradient Accumulation Step
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                if use_amp and cfg.training.precision == "fp16":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent exploding gradients
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                optimizer.zero_grad()
                global_step += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item() * cfg.training.gradient_accumulation_steps:.4f}"})

        # End of Epoch Checkpointing
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
        
        ckpt_path = ckpt_dir / f"axiomllm_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    train()