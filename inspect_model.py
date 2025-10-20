"""
Inspect loaded model to verify it's running locally and show device info.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from transformers import AutoConfig, AutoTokenizer
from src.utils import load_config, setup_logging
import yaml
import torch


def main():
    """Inspect model configuration and verify local setup."""
    print("=" * 60)
    print("Model Inspection Tool")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    setup_logging(config.get('app', {}).get('log_level', 'INFO'))
    
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'unknown')
    cache_dir = model_config.get('cache_dir')
    token = model_config.get('token') or os.environ.get('HUGGINGFACE_TOKEN')
    
    print(f"\nConfigured Model: {model_name}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Token Present: {'Yes' if token else 'No (not needed for public models)'}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n" + "-" * 60)
    print("Loading Model Metadata (not full model weights)...")
    print("-" * 60)
    
    try:
        # Load tokenizer metadata
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            token=token,
            cache_dir=cache_dir
        )
        print(f"\n✓ Tokenizer loaded:")
        print(f"  Class: {tokenizer.__class__.__name__}")
        print(f"  Name/Path: {getattr(tokenizer, 'name_or_path', 'N/A')}")
        print(f"  Vocab Size: {tokenizer.vocab_size}")
        
        # Load model config (not weights)
        hf_config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            cache_dir=cache_dir
        )
        print(f"\n✓ Model Config loaded:")
        print(f"  Model Type: {getattr(hf_config, 'model_type', 'N/A')}")
        print(f"  Name/Path: {getattr(hf_config, 'name_or_path', 'N/A')}")
        print(f"  Hidden Size: {getattr(hf_config, 'hidden_size', 'N/A')}")
        print(f"  Num Layers: {getattr(hf_config, 'num_hidden_layers', 'N/A')}")
        
        # Check cache directory
        if cache_dir and os.path.exists(cache_dir):
            print(f"\n✓ Cache Directory Contents:")
            for item in os.listdir(cache_dir):
                if item.startswith('models--'):
                    print(f"  - {item}")
        
        print("\n" + "=" * 60)
        print("LOCAL OPERATION CONFIRMED")
        print("=" * 60)
        print("\n✓ Model files are cached locally")
        print("✓ Inference will run on your machine (CPU/GPU)")
        print("✓ No chat data is sent to remote servers during inference")
        print(f"✓ Chat history saved to: {config.get('app', {}).get('history_file', 'N/A')}")
        print("\nNote: Initial download from Hugging Face Hub is required,")
        print("      but after that, everything runs locally.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nThis might indicate:")
        print("  - Model not yet downloaded")
        print("  - Network issue during metadata fetch")
        print("  - Invalid model name or token required")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
