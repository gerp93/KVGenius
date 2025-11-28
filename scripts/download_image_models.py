"""
Download all image generation models with resume support.
Reads models from config/image_model_presets.yaml
Uses single-threaded downloads to avoid timeout issues.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fix_dll_paths  # Must be before torch imports

# Disable Xet storage - use direct HTTP (more reliable)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import yaml
from huggingface_hub import snapshot_download

CACHE_DIR = "./data/model_cache"

def load_models_from_config():
    """Load model list from user's config file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "image_model_presets.yaml")
    example_path = os.path.join(os.path.dirname(__file__), "..", "config", "image_model_presets.example.yaml")
    
    # Try user config first, fall back to example
    if os.path.exists(config_path):
        path_to_use = config_path
    elif os.path.exists(example_path):
        path_to_use = example_path
        print("⚠️ No user config found, using example config")
    else:
        print("❌ No config file found!")
        return {}
    
    try:
        with open(path_to_use, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            models = config.get('models', {})
            # Convert to {name: id} format
            return {name: preset.get('id') for name, preset in models.items() if preset.get('id')}
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {}

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
    # Load models from config
    IMAGE_MODELS = load_models_from_config()
    
    if not IMAGE_MODELS:
        print("No models found in config. Please check your image_model_presets.yaml")
        return
    
    print("="*60)
    print("Image Model Downloader")
    print("="*60)
    print(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    print(f"Models to download: {len(IMAGE_MODELS)}")
    print()
    for name, repo_id in IMAGE_MODELS.items():
        print(f"  • {name}: {repo_id}")
    
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
