
import os
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logger = logging.getLogger(__name__)

class AxiomTokenizer:
    def __init__(self, vocab_size: int = 50257, save_path: str = "assets/axiom_tokenizer.json"):
        self.vocab_size = vocab_size
        self.save_path = save_path
        self._tokenizer: Tokenizer | None = None

    def train_from_files(self, file_paths: list[str]) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        
        special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)
        
        tokenizer.train(file_paths, trainer)
        tokenizer.save(self.save_path)
        self._tokenizer = tokenizer
        logger.info(f"Tokenizer trained and saved to {self.save_path}")

    def load(self) -> None:
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Tokenizer not found at {self.save_path}")
        self._tokenizer = Tokenizer.from_file(self.save_path)
        logger.info(f"Loaded tokenizer from {self.save_path}")

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        if self._tokenizer is None:
            self.load()
        ids = self._tokenizer.encode(text).ids
        if add_eos:
            ids.append(self._tokenizer.token_to_id("<eos>"))
        return ids

    def decode(self, ids: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")
        return self._tokenizer.decode(ids, skip_special_tokens=True)