import torch
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.model import AxiomLLM
from src.tokenizer import AxiomTokenizer

def chat():
    # 1. Load Config & Device
    cfg = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AxiomLLM on {device}...")

    # 2. Load Tokenizer
    tokenizer = AxiomTokenizer(vocab_size=cfg.model.vocab_size, save_path="assets/axiom_tokenizer.json")
    tokenizer.load()

    # 3. Initialize Model & Load Weights
    model = AxiomLLM(cfg.model).to(device)
    
    ckpt_path = "checkpoints/axiomllm_epoch_3.pt"
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Handle cases where state_dict might be nested
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode (disables dropout, etc.)
        print(f"✅ Loaded weights from {ckpt_path}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {ckpt_path}. Did you download it from Drive?")
        return

    # 4. The Generation Loop
    print("\n" + "="*40)
    print("  AXIOMLLM CHAT (Type 'quit' to exit)")
    print("="*40)
    
    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
            
        # Tokenize Input
        input_ids = tokenizer.encode(prompt, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        generated_tokens = input_ids.copy()
        max_new_tokens = 50 # Keep it short for the smoke test
        
        print("Axiom: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward Pass (Returns logits, aux_loss)
                logits, _ = model(input_tensor)
                
                # Get logits for the *last* predicted token
                next_token_logits = logits[0, -1, :]
                
                # Temperature Scaling (Controls randomness. 0.8 is balanced.)
                # Higher (>1.0) = more creative/crazy. Lower (<0.5) = more repetitive/safe.
                next_token_logits = next_token_logits / 0.8
                
                # Softmax -> Probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Top-K Filtering (Optional: Keep only top 50 likely words to avoid garbage)
                # For this smoke test, we just sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                next_token_id = next_token.item()
                generated_tokens.append(next_token_id)
                
                # Update input tensor for next iteration
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                # Decode and print streaming
                text = tokenizer.decode([next_token_id])
                print(text, end="", flush=True)
                
                # Stop condition
                if next_token_id == tokenizer._tokenizer.token_to_id("<eos>"):
                    break
                    
        print() # Newline after generation

if __name__ == "__main__":
    chat()