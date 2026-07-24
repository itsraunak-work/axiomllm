import os, sys, torch, torch.nn.functional as F, logging
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.utils import setup_logging, seed_everything, get_device, ensure_dir
from src.tokenizer import AxiomTokenizer
from src.data import prepare_dataloader
from src.model import AxiomLLM

logger = logging.getLogger(__name__)

def train():
    setup_logging("INFO")
    cfg = load_config("configs/default.yaml")
    seed_everything(cfg.training.seed)
    device = get_device()
    ensure_dir(cfg.training.checkpoint_dir)

    tokenizer = AxiomTokenizer(vocab_size=cfg.model.vocab_size)
    dummy_file = "assets/dummy_corpus.txt"
    if not os.path.exists("assets/axiom_tokenizer.json"):
        os.makedirs("assets", exist_ok=True)
        with open(dummy_file, "w", encoding="utf-8") as f: f.write("The quick brown fox jumps over the lazy dog. " * 5000)
        tokenizer.train_from_files([dummy_file])
    else: tokenizer.load()

    dataloader = prepare_dataloader(cfg.training.dataset_name, cfg.training.dataset_split, tokenizer, cfg.model.max_seq_len, cfg.training.batch_size, max_samples=5000)
    
    model = AxiomLLM(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    model.train()
    for epoch in range(cfg.training.max_epochs):
        optimizer.zero_grad()
        for step, (input_ids, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits, aux_loss = model(input_ids)
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = (ce_loss + 0.01 * aux_loss) / cfg.training.gradient_accumulation_steps
            
            if use_amp and amp_dtype == torch.float16: scaler.scale(loss).backward()
            else: loss.backward()
                
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                optimizer.zero_grad()

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, Path(cfg.training.checkpoint_dir) / f"axiomllm_epoch_{epoch+1}.pt")
        logger.info(f"Epoch {epoch+1} Complete | Checkpoint Saved")

if __name__ == "__main__":
    train()