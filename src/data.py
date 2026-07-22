
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.tokenizer import AxiomTokenizer

logger = logging.getLogger(__name__)

class PackedDataset(Dataset):
    """
    Instead of padding short sentences (which wastes GPU compute), 
    we concatenate all text into one giant stream and slice it into fixed-length chunks.
    """
    def __init__(self, token_ids: list[int], seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        
        self.num_sequences = (len(self.token_ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.token_ids[start:end]
        
       
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels

def prepare_dataloader(
    dataset_name: str,
    split: str,
    tokenizer: AxiomTokenizer,
    seq_len: int,
    batch_size: int,
    max_samples: int = 10000
) -> DataLoader:
    # streaming=True prevents downloading the whole dataset to disk/RAM
    raw_dataset = load_dataset(dataset_name, split=split, streaming=True)
    token_buffer = []
    count = 0

    logger.info(f"Streaming and tokenizing up to {max_samples} samples...")
    for sample in raw_dataset:
        # TinyStories uses "story", other datasets use "text"
        text = sample.get("story", sample.get("text", ""))
        if not text or count >= max_samples:
            break
        
        token_buffer.extend(tokenizer.encode(text, add_eos=True))
        count += 1

    logger.info(f"Total tokens buffered: {len(token_buffer)}")
    
    dataset = PackedDataset(token_buffer, seq_len)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for Windows compatibility with streaming
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    logger.info(f"DataLoader ready: {len(dataset)} sequences of length {seq_len}")
    return loader