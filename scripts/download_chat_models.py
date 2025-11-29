"""
Download all chat models from config with resume support.
Reads models from config/chat_model_presets.yaml
Skips models that are already downloaded (unless --force is used).
Uses single-threaded downloads to avoid timeout issues.

Usage:
  python download_chat_models.py          # Download missing models only
  python download_chat_models.py --force  # Re-download all models
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fix_dll_paths  # Must be before torch imports

# Disable Xet storage - use direct HTTP (more reliable)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import yaml
from huggingface_hub import snapshot_download, scan_cache_dir

CACHE_DIR = "./data/model_cache"


def load_models_from_config():
    """Load model list from user's config file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "chat_model_presets.yaml")
    example_path = os.path.join(os.path.dirname(__file__), "..", "config", "chat_model_presets.example.yaml")
    
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
            # Return full model info
            return {name: preset for name, preset in models.items() if preset.get('id')}
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {}


def get_downloaded_models():
    """Get set of already downloaded model repo IDs."""
    downloaded = set()
    
    if not os.path.exists(CACHE_DIR):
        return downloaded
    
    try:
        cache_info = scan_cache_dir(CACHE_DIR)
        for repo in cache_info.repos:
            # repo.repo_id is like "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
            downloaded.add(repo.repo_id)
    except Exception as e:
        print(f"⚠️ Could not scan cache: {e}")
        # Fallback: check directory names
        models_dir = os.path.join(CACHE_DIR, "models--*")
        import glob
        for path in glob.glob(os.path.join(CACHE_DIR, "models--*")):
            # Convert "models--NousResearch--Nous-Hermes-2-Mistral-7B-DPO" to "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
            dirname = os.path.basename(path)
            if dirname.startswith("models--"):
                repo_id = dirname[8:].replace("--", "/", 1)
                downloaded.add(repo_id)
    
    return downloaded


def download_model(name: str, repo_id: str, description: str = ""):
    """Download a single model with resume support."""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Repository: {repo_id}")
    if description:
        print(f"Description: {description}")
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
    # Parse arguments
    parser = argparse.ArgumentParser(description="Download all chat models from config")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-download all models, even if already cached")
    args = parser.parse_args()
    
    # Load models from config
    CHAT_MODELS = load_models_from_config()
    
    if not CHAT_MODELS:
        print("No models found in config. Please check your chat_model_presets.yaml")
        return
    
    # Get already downloaded models (unless forcing)
    if args.force:
        downloaded = set()  # Ignore cache when forcing
        print("⚠️ Force mode: will re-download all models")
    else:
        downloaded = get_downloaded_models()
    
    print("="*60)
    print("Chat Model Downloader")
    print("="*60)
    print(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    print(f"Models in config: {len(CHAT_MODELS)}")
    if not args.force:
        print(f"Already downloaded: {len(downloaded)}")
    print()
    
    # Categorize models
    to_download = {}
    already_have = {}
    
    for name, preset in CHAT_MODELS.items():
        repo_id = preset.get('id')
        if repo_id in downloaded:
            already_have[name] = preset
        else:
            to_download[name] = preset
    
    # Show status
    if already_have and not args.force:
        print("✅ Already downloaded:")
        for name, preset in already_have.items():
            print(f"   • {name} ({preset.get('id')})")
        print()
    
    if not to_download:
        print("🎉 All models are already downloaded!")
        if not args.force:
            print("💡 Use --force to re-download anyway")
        return
    
    print(f"📥 Models to download ({len(to_download)}):")
    for name, preset in to_download.items():
        desc = preset.get('description', '')
        vram = preset.get('vram', '')
        print(f"   • {name}: {preset.get('id')}")
        if desc:
            print(f"     {desc}")
        if vram:
            print(f"     VRAM: {vram}")
    
    print()
    response = input("Download these models? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    results = {}
    for name, preset in to_download.items():
        repo_id = preset.get('id')
        description = preset.get('description', '')
        results[name] = download_model(name, repo_id, description)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    success = sum(1 for v in results.values() if v)
    failed = len(results) - success
    
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {name}")
    
    print(f"\nDownloaded: {success} succeeded, {failed} failed")
    print(f"Previously had: {len(already_have)} models")
    print(f"Total available: {success + len(already_have)} models")
    
    if failed > 0:
        print("\nTip: Run this script again to retry failed downloads.")
        print("Downloads will resume from where they left off.")


if __name__ == "__main__":
    main()
