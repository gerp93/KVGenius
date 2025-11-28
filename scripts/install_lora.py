"""
Install a pre-built LoRA .safetensors file for use in KVGenius.

Usage:
    python scripts/install_lora.py <safetensors_file> <lora_name> [trigger_word]

Example:
    python scripts/install_lora.py "C:\Downloads\twilek_lora.safetensors" "Twilek" "twilek"
"""

import sys
import os
import shutil
import json
from pathlib import Path

LORA_DIR = Path("data/lora_models")


def get_lora_rank_from_file(safetensors_path):
    """Try to detect LoRA rank from the safetensors file."""
    try:
        from safetensors import safe_open
        
        with safe_open(safetensors_path, framework="pt") as f:
            keys = f.keys()
            # Look for a lora_down weight to determine rank
            for key in keys:
                if "lora_down" in key.lower() or "lora_a" in key.lower():
                    tensor = f.get_tensor(key)
                    # Rank is typically the smaller dimension
                    rank = min(tensor.shape)
                    print(f"  Detected LoRA rank: {rank}")
                    return rank
    except Exception as e:
        print(f"  Could not detect rank: {e}")
    
    return 16  # Default rank


def install_lora(safetensors_path: str, lora_name: str, trigger_word: str = None):
    """Install a LoRA .safetensors file."""
    
    safetensors_path = Path(safetensors_path)
    
    if not safetensors_path.exists():
        print(f"❌ File not found: {safetensors_path}")
        return False
    
    if not safetensors_path.suffix.lower() == ".safetensors":
        print(f"❌ File must be a .safetensors file")
        return False
    
    # Create LoRA directory
    lora_dir = LORA_DIR / lora_name
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Installing LoRA: {lora_name}")
    print(f"  Source: {safetensors_path}")
    print(f"  Destination: {lora_dir}")
    
    # Copy the safetensors file
    dest_file = lora_dir / "adapter_model.safetensors"
    shutil.copy2(safetensors_path, dest_file)
    print(f"  ✓ Copied weights")
    
    # Detect rank
    rank = get_lora_rank_from_file(dest_file)
    
    # Create adapter_config.json (required by PEFT/diffusers)
    adapter_config = {
        "base_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",  # Generic, works for most
        "inference_mode": True,
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank * 2,  # Common convention
        "lora_dropout": 0.0,
        "target_modules": [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "conv1", "conv2", "conv_shortcut",
            "downsamplers.0.conv", "upsamplers.0.conv"
        ],
        "modules_to_save": None,
        "fan_in_fan_out": False,
        "bias": "none"
    }
    
    config_path = lora_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"  ✓ Created adapter_config.json")
    
    # Create metadata file
    if trigger_word is None:
        trigger_word = lora_name.lower().replace(" ", "_")
    
    metadata = {
        "name": lora_name,
        "trigger_word": trigger_word,
        "source": str(safetensors_path.name),
        "installed_from": "external",
        "rank": rank,
        "description": f"Pre-built LoRA installed from {safetensors_path.name}"
    }
    
    metadata_path = lora_dir / "lora_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Created lora_metadata.json")
    
    print(f"\n✅ LoRA '{lora_name}' installed successfully!")
    print(f"\n📝 Trigger word: {trigger_word}")
    print(f"   Use this word in your prompt to activate the LoRA")
    
    return True


def list_installed_loras():
    """List all installed LoRAs."""
    if not LORA_DIR.exists():
        print("No LoRAs installed yet.")
        return
    
    loras = [d for d in LORA_DIR.iterdir() if d.is_dir()]
    
    if not loras:
        print("No LoRAs installed yet.")
        return
    
    print(f"\n📦 Installed LoRAs ({len(loras)}):")
    print("-" * 50)
    
    for lora_path in sorted(loras):
        metadata_file = lora_path / "lora_metadata.json"
        weights_file = lora_path / "adapter_model.safetensors"
        
        name = lora_path.name
        trigger = "unknown"
        size_mb = 0
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    meta = json.load(f)
                    trigger = meta.get("trigger_word", "unknown")
            except:
                pass
        
        if weights_file.exists():
            size_mb = weights_file.stat().st_size / (1024 * 1024)
        
        print(f"  • {name}")
        print(f"    Trigger: {trigger} | Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n" + "=" * 50)
        list_installed_loras()
        sys.exit(0)
    
    if sys.argv[1] in ["--list", "-l", "list"]:
        list_installed_loras()
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print("❌ Please provide both the .safetensors file path and a name for the LoRA")
        print("\nUsage: python scripts/install_lora.py <file.safetensors> <name> [trigger_word]")
        sys.exit(1)
    
    safetensors_file = sys.argv[1]
    lora_name = sys.argv[2]
    trigger_word = sys.argv[3] if len(sys.argv) > 3 else None
    
    success = install_lora(safetensors_file, lora_name, trigger_word)
    sys.exit(0 if success else 1)
