"""
Download a single model interactively.
Shows all configured models and lets you pick one to download.
Skips models that are already downloaded.

Usage:
  python download_model.py                     # Interactive mode
  python download_model.py --list              # List all models
  python download_model.py --download 5        # Download model #5
  python download_model.py --download 5 --force  # Force re-download model #5
"""
import os
import sys
import glob
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fix_dll_paths  # Must be before torch imports

# Disable Xet storage - use direct HTTP (more reliable)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import yaml
from huggingface_hub import snapshot_download, scan_cache_dir

CACHE_DIR = "./data/model_cache"


def load_all_models():
    """Load models from both chat and image config files."""
    models = {}
    
    config_files = [
        ("Chat", "chat_model_presets.yaml", "chat_model_presets.example.yaml"),
        ("Image", "image_model_presets.yaml", "image_model_presets.example.yaml"),
    ]
    
    for model_type, config_name, example_name in config_files:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", config_name)
        example_path = os.path.join(os.path.dirname(__file__), "..", "config", example_name)
        
        path_to_use = config_path if os.path.exists(config_path) else example_path
        
        if os.path.exists(path_to_use):
            try:
                with open(path_to_use, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    for name, preset in config.get('models', {}).items():
                        if preset.get('id'):
                            models[f"{model_type}: {name}"] = preset
            except Exception as e:
                print(f"⚠️ Error loading {config_name}: {e}")
    
    return models


def get_downloaded_models():
    """Get set of already downloaded model repo IDs."""
    downloaded = set()
    
    if not os.path.exists(CACHE_DIR):
        return downloaded
    
    try:
        cache_info = scan_cache_dir(CACHE_DIR)
        for repo in cache_info.repos:
            downloaded.add(repo.repo_id)
    except Exception:
        # Fallback: check directory names
        for path in glob.glob(os.path.join(CACHE_DIR, "models--*")):
            dirname = os.path.basename(path)
            if dirname.startswith("models--"):
                repo_id = dirname[8:].replace("--", "/", 1)
                downloaded.add(repo_id)
    
    return downloaded


def download_model(name: str, repo_id: str, force: bool = False):
    """Download a single model with resume support."""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Repository: {repo_id}")
    if force:
        print("⚠️ FORCE MODE: Will re-download even if files exist")
    print("This may take a while. Using single-threaded download to avoid timeouts.")
    print(f"{'='*60}")
    
    try:
        path = snapshot_download(
            repo_id,
            cache_dir=CACHE_DIR,
            max_workers=1,
            resume_download=True,
            force_download=force,  # Force re-download if requested
        )
        print(f"\n✅ Downloaded successfully!")
        print(f"   Path: {path}")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return False


def main():
    # Load all models
    all_models = load_all_models()
    
    if not all_models:
        print("❌ No models found in config files.")
        return
    
    # Get already downloaded
    downloaded = get_downloaded_models()
    
    print("="*60)
    print("Single Model Downloader")
    print("="*60)
    print(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    print()
    
    # Build numbered list
    model_list = list(all_models.items())
    
    print("Available models:\n")
    for i, (name, preset) in enumerate(model_list, 1):
        repo_id = preset.get('id')
        is_downloaded = repo_id in downloaded
        status = "✅" if is_downloaded else "  "
        desc = preset.get('description', '')
        vram = preset.get('vram', '')
        
        print(f"  {status} [{i:2d}] {name}")
        print(f"         {repo_id}")
        if desc:
            print(f"         {desc}")
        if vram:
            print(f"         VRAM: {vram}")
        print()
    
    print("="*60)
    print("✅ = Already downloaded")
    print()
    
    # Get user choice
    try:
        choice = input("Enter number to download (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("Cancelled.")
            return
        
        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_list):
            print("❌ Invalid selection.")
            return
        
        name, preset = model_list[idx]
        repo_id = preset.get('id')
        
        # Check if already downloaded
        force = False
        if repo_id in downloaded:
            print(f"\n⚠️ {name} is already downloaded!")
            response = input("Force re-download? [y/N]: ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
            force = True  # User wants to force re-download
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        download_model(name, repo_id, force=force)
        
    except ValueError:
        print("❌ Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nCancelled.")


if __name__ == "__main__":
    main()
