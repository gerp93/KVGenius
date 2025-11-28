"""
Download LoRA from CivitAI and install it to data/lora_models folder.

Usage:
    python download_civitai_lora.py <civitai_model_id> <lora_name>
    
Example:
    python download_civitai_lora.py 123456 "Twilek_CivitAI"
    
To find the model ID, go to the CivitAI model page.
The ID is in the URL: https://civitai.com/models/123456/model-name
"""

import sys
import requests
import json
from pathlib import Path

LORA_DIR = Path("data/lora_models")


def get_civitai_download_url(model_id: str) -> tuple:
    """Get download URL and info from CivitAI API."""
    # Get model info
    api_url = f"https://civitai.com/api/v1/models/{model_id}"
    print(f"Fetching: {api_url}")
    
    response = requests.get(api_url, headers={"User-Agent": "KVGenius/1.0"})
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch model info: {response.status_code}")
    
    data = response.json()
    
    # Get the latest version
    if not data.get("modelVersions"):
        raise Exception("No versions found for this model")
    
    latest_version = data["modelVersions"][0]
    
    # Find the safetensors file
    for file in latest_version.get("files", []):
        if file["name"].endswith(".safetensors"):
            return file["downloadUrl"], {
                "name": data["name"],
                "description": data.get("description", ""),
                "trigger_words": latest_version.get("trainedWords", []),
                "base_model": latest_version.get("baseModel", "SD 1.5"),
            }
    
    raise Exception("No safetensors file found")


def download_lora(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress."""
    print(f"Downloading from: {url}")
    
    response = requests.get(url, stream=True, headers={"User-Agent": "KVGenius/1.0"})
    
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    
    total_size = int(response.headers.get('content-length', 0))
    
    downloaded = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    mb = downloaded / 1024 / 1024
                    total_mb = total_size / 1024 / 1024
                    print(f"\r  Progress: {pct:.1f}% ({mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)
    
    print(f"\n  ✓ Downloaded: {output_path}")


def install_lora(model_id: str, lora_name: str):
    """Download and install a LoRA from CivitAI."""
    print(f"\n{'='*60}")
    print(f"Installing LoRA: {lora_name}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = LORA_DIR / lora_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get download URL
    print("Fetching model info from CivitAI...")
    download_url, info = get_civitai_download_url(model_id)
    
    print(f"  Name: {info['name']}")
    print(f"  Base Model: {info['base_model']}")
    trigger_words = info['trigger_words']
    print(f"  Trigger Words: {', '.join(trigger_words) if trigger_words else 'None specified'}")
    print()
    
    # Download the safetensors file
    safetensors_path = output_dir / "adapter_model.safetensors"
    download_lora(download_url, safetensors_path)
    
    # Determine trigger word
    trigger_word = trigger_words[0] if trigger_words else lora_name.lower().replace("_", " ")
    
    # Create metadata file
    metadata = {
        "name": lora_name,
        "trigger_word": trigger_word,
        "base_model": info["base_model"],
        "source": f"CivitAI Model {model_id}",
        "original_name": info["name"],
        "all_trigger_words": trigger_words,
    }
    
    metadata_path = output_dir / "lora_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create a minimal adapter_config.json for compatibility
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": info["base_model"],
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
    }
    
    config_path = output_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ LoRA installed successfully!")
    print(f"  Location: {output_dir}")
    print(f"  Trigger word: {trigger_word}")
    if len(trigger_words) > 1:
        print(f"  Other triggers: {', '.join(trigger_words[1:])}")
    print(f"{'='*60}\n")
    
    return output_dir


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nHow to find CivitAI Model ID:")
        print("  1. Go to https://civitai.com/models?query=twilek")
        print("  2. Click on a LoRA you want")
        print("  3. The URL will be like: https://civitai.com/models/123456/lora-name")
        print("  4. The model ID is: 123456")
        print()
        print("Example:")
        print('  python download_civitai_lora.py 123456 "Twilek_CivitAI"')
        sys.exit(1)
    
    model_id = sys.argv[1]
    lora_name = sys.argv[2].replace(" ", "_")  # Replace spaces with underscores
    
    try:
        install_lora(model_id, lora_name)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
