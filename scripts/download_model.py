"""Download model with resume support and single-threaded to avoid timeouts."""
from huggingface_hub import snapshot_download

repo_id = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo"

print(f"Downloading {repo_id} with resume support...")
print("This may take a while. Using single-threaded download to avoid timeouts.")

path = snapshot_download(
    repo_id,
    cache_dir="data/model_cache",
    resume_download=True,
    max_workers=1,
    force_download=False,
)

print(f"✅ Downloaded to: {path}")
