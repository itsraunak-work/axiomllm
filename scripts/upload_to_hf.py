import os
from huggingface_hub import HfApi, create_repo

# The HfApi automatically reads the secure token from your OS cache!
api = HfApi()
REPO_ID = "itsraunak-work/axiomllm" 

print(f"Connecting to Hugging Face as itsraunak-work...")
print(f"Creating repo: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo status: {e}")

print("Uploading Tokenizer (Fast)...")
api.upload_file(
    path_or_fileobj="assets/axiom_tokenizer.json",
    path_in_repo="tokenizer.json",
    repo_id=REPO_ID,
    repo_type="model",
)

print("Uploading 2GB Model Weights (This will take a few minutes)...")
print("Do not close this window!")
api.upload_file(
    path_or_fileobj="checkpoints/axiomllm_epoch_3.pt",
    path_in_repo="model.pt",
    repo_id=REPO_ID,
    repo_type="model",
)

print("="*50)
print(f"✅ SUCCESS! Your model is live at:")
print(f"https://huggingface.co/{REPO_ID}")
print("="*50)