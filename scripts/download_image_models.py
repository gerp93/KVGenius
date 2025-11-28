"""
Download all image generation models with resume support.
Uses single-threaded downloads to avoid timeout issues.
"""
import fix_dll_paths  # Must be first

import os
import sys

# Disable Xet storage - use direct HTTP (more reliable)
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

# Models to download - SMALLEST FIRST for quick results
IMAGE_MODELS = {
    "Dreamshaper 8": "Lykon/dreamshaper-8",  # ~2 GB - great quality
    "AbsoluteReality": "Lykon/absolute-reality-1.6525",  # ~2 GB - photorealistic  
    "Realistic Vision V5": "SG161222/Realistic_Vision_V5.1_noVAE",  # ~2 GB
    "Stable Diffusion 1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",  # ~4 GB
    "SDXL-Turbo": "stabilityai/sdxl-turbo",  # ~10 GB - last (largest)
}

CACHE_DIR = "./data/model_cache"

def download_model(name: str, repo_id: str):
    """Download a single model with resume support."""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}")
    
    try:
        path = snapshot_download(
            repo_id,
            cache_dir=CACHE_DIR,
            max_workers=1,  # Single-threaded to avoid timeouts
            resume_download=True,
        )
        print(f"✅ {name} downloaded successfully!")
        print(f"   Path: {path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        return False

def main():
    print("="*60)
    print("Image Model Downloader")
    print("="*60)
    print(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    print(f"Models to download: {len(IMAGE_MODELS)}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    results = {}
    for name, repo_id in IMAGE_MODELS.items():
        results[name] = download_model(name, repo_id)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    success = sum(1 for v in results.values() if v)
    failed = len(results) - success
    
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {name}")
    
    print(f"\nTotal: {success} succeeded, {failed} failed")
    
    if failed > 0:
        print("\nTip: Run this script again to retry failed downloads.")
        print("Downloads will resume from where they left off.")

if __name__ == "__main__":
    main()
