"""
Core image generation engine for KVGenius.
UI-agnostic - can be used by Gradio, Flet, or CLI.
"""
import os
import sys
import time
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass

# Ensure fix_dll_paths is imported before torch
sys.path.insert(0, str(Path(__file__).parent.parent))
import fix_dll_paths

import torch
from PIL import Image

from .config import (
    CACHE_DIR, 
    GENERATED_IMAGES_DIR,
    HUGGINGFACE_MODELS, 
    LOCAL_CHECKPOINTS,
    CHECKPOINTS_DIR,
    LORA_DIR,
)

logger = logging.getLogger(__name__)

# Global state
_current_model: Optional[Any] = None
_current_model_key: Optional[str] = None
_loaded_models: Dict[str, Any] = {}
_current_lora: Optional[str] = None  # Track loaded LoRA


@dataclass
class GenerationResult:
    """Result of image generation."""
    image: Image.Image
    seed: int
    generation_time: float
    filepath: Optional[str] = None
    base64_data: Optional[str] = None


def get_available_image_models() -> Dict[str, Dict[str, Any]]:
    """Get all available image models (HuggingFace + Local)."""
    models = {}
    
    # Add HuggingFace models
    for name, preset in HUGGINGFACE_MODELS.items():
        models[name] = {
            "id": preset.get("id", ""),
            "type": preset.get("type", "sd"),
            "source": "huggingface",
            "description": preset.get("description", ""),
            "vram": preset.get("vram", "~6 GB"),
            "default_steps": preset.get("default_steps", 25),
            "default_guidance": preset.get("default_guidance", 7.0),
            "default_width": preset.get("default_width", 512),
            "default_height": preset.get("default_height", 512),
            "step_range": preset.get("step_range", [10, 50]),
            "guidance_range": preset.get("guidance_range", [1.0, 20.0]),
            "default_negative": preset.get("default_negative", ""),
        }
    
    # Add local checkpoints
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.iterdir():
            if f.suffix.lower() in ['.safetensors', '.ckpt']:
                size_gb = f.stat().st_size / (1024**3)
                is_sdxl = size_gb > 4.0 or "xl" in f.stem.lower()
                models[f.stem] = {
                    "id": str(f),
                    "type": "sdxl" if is_sdxl else "sd",
                    "source": "local",
                    "description": f"Local checkpoint ({size_gb:.1f} GB)",
                    "vram": f"~{4 if not is_sdxl else 8} GB",
                    "default_steps": 25,
                    "default_guidance": 7.0,
                    "default_width": 1024 if is_sdxl else 512,
                    "default_height": 1024 if is_sdxl else 512,
                    "is_local": True,
                }
    
    return models


def is_model_downloaded(repo_id: str) -> bool:
    """Check if a HuggingFace model is downloaded."""
    cache_name = "models--" + repo_id.replace("/", "--")
    cache_path = CACHE_DIR / cache_name
    
    if not cache_path.exists():
        return False
    
    # Check for substantial files (>100MB)
    min_size = 100 * 1024 * 1024
    total_size = 0
    
    for root, _, files in os.walk(cache_path):
        for fname in files:
            try:
                total_size += os.path.getsize(os.path.join(root, fname))
                if total_size >= min_size:
                    return True
            except:
                pass
    
    return False


def get_current_model() -> Tuple[Optional[Any], Optional[str]]:
    """Get the currently loaded model and its key."""
    return _current_model, _current_model_key


def _infer_lora_model_type(base_model: str, name: str) -> str:
    """Infer the model type (sd, sdxl, flux) from base_model or LoRA name."""
    if not base_model:
        # Try to infer from name
        name_lower = name.lower()
        if "sdxl" in name_lower or "xl" in name_lower:
            return "sdxl"
        elif "flux" in name_lower:
            return "flux"
        return "sd"  # Default to SD 1.5
    
    base_lower = base_model.lower()
    if "sdxl" in base_lower or "xl-" in base_lower:
        return "sdxl"
    elif "flux" in base_lower:
        return "flux"
    elif any(x in base_lower for x in ["sd-1", "sd1", "1.5", "realistic_vision", "dreamshaper", "epicrealism"]):
        return "sd"
    return "sd"  # Default


def get_available_loras(model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get list of available LoRA models from the lora_models directory.
    
    Args:
        model_type: Optional filter - 'sd', 'sdxl', 'flux', or None for all
    """
    loras = []
    
    if not LORA_DIR.exists():
        return loras
    
    for item in LORA_DIR.iterdir():
        if item.is_dir():
            # Check for adapter files
            has_adapter = (item / "adapter_model.safetensors").exists() or \
                         (item / "adapter_model.bin").exists()
            has_config = (item / "adapter_config.json").exists()
            
            # Check for loose safetensors
            safetensors = list(item.glob("*.safetensors"))
            
            if has_adapter or has_config or safetensors:
                # Try to read metadata
                metadata_file = item / "lora_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        import json
                        metadata = json.loads(metadata_file.read_text())
                    except:
                        pass
                
                # Get base_model and infer type
                base_model = metadata.get("base_model", "")
                lora_type = _infer_lora_model_type(base_model, item.name)
                
                # Apply filter if specified
                if model_type and lora_type != model_type:
                    continue
                
                loras.append({
                    "name": item.name,
                    "path": str(item),
                    "description": metadata.get("description", ""),
                    "trigger_words": metadata.get("trigger_words", metadata.get("trigger_word", [])),
                    "has_adapter": has_adapter,
                    "base_model": base_model,
                    "model_type": lora_type,
                })
        elif item.suffix == ".safetensors" and item.stem not in ["adapter_model"]:
            # Loose safetensors file - try to infer type from name
            lora_type = _infer_lora_model_type("", item.stem)
            
            if model_type and lora_type != model_type:
                continue
            
            loras.append({
                "name": item.stem,
                "path": str(item),
                "description": "Single LoRA file",
                "trigger_words": [],
                "has_adapter": False,
                "base_model": "",
                "model_type": lora_type,
            })
    
    return loras


def unload_lora():
    """Unload any currently loaded LoRA."""
    global _current_model, _current_lora
    
    if _current_model is not None and _current_lora is not None:
        try:
            if hasattr(_current_model, 'unfuse_lora'):
                _current_model.unfuse_lora()
            if hasattr(_current_model, 'unload_lora_weights'):
                _current_model.unload_lora_weights()
            logger.info(f"Unloaded LoRA: {_current_lora}")
        except Exception as e:
            logger.warning(f"Error unloading LoRA: {e}")
    
    _current_lora = None


def unload_image_model():
    """Unload current image model to free VRAM."""
    global _current_model, _current_model_key, _loaded_models
    
    if _current_model is not None:
        logger.info(f"Unloading image model: {_current_model_key}")
        del _current_model
        _current_model = None
        _current_model_key = None
    
    _loaded_models.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_image_model(
    model_key: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Load an image model.
    
    Args:
        model_key: Name of the model to load
        progress_callback: Optional callback(progress: 0-1, message: str)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    global _current_model, _current_model_key, _loaded_models
    
    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct*100:.0f}%] {msg}")
    
    models = get_available_image_models()
    if model_key not in models:
        return False, f"Unknown model: {model_key}"
    
    model_info = models[model_key]
    model_id = model_info["id"]
    model_type = model_info.get("type", "sd")
    is_local = model_info.get("is_local", False)
    
    # Check if already loaded
    if _current_model_key == model_key and _current_model is not None:
        return True, f"{model_key} already loaded"
    
    # Unload previous model
    if _current_model is not None:
        report_progress(0.05, "Unloading previous model...")
        unload_image_model()
    
    report_progress(0.1, f"Loading {model_key}...")
    
    # Auto-detect SDXL from model ID if type not explicitly set
    if model_type == "sd" and ("sdxl" in model_id.lower() or "xl" in model_id.lower()):
        model_type = "sdxl"
        logger.info(f"Auto-detected SDXL pipeline for {model_key}")
    
    try:
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            AutoPipelineForText2Image,
        )
        
        # Determine pipeline class based on model type
        if model_type == "flux":
            from diffusers import FluxPipeline
            report_progress(0.2, "Loading Flux pipeline...")
            pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=str(CACHE_DIR),
            )
        elif model_type == "sd3":
            from diffusers import StableDiffusion3Pipeline
            report_progress(0.2, "Loading SD3 pipeline...")
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=str(CACHE_DIR),
            )
        elif is_local:
            report_progress(0.2, "Loading local checkpoint...")
            # Determine if SDXL
            is_sdxl = model_type == "sdxl"
            if is_sdxl:
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_id,
                    torch_dtype=torch.float16,
                )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    model_id,
                    torch_dtype=torch.float16,
                )
        elif model_type in ["sdxl", "sdxl-lightning"]:
            report_progress(0.2, "Loading SDXL pipeline...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=str(CACHE_DIR),
                use_safetensors=True,
            )
        else:
            # Standard SD 1.5
            report_progress(0.2, "Loading SD pipeline...")
            try:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    cache_dir=str(CACHE_DIR),
                    use_safetensors=True,
                )
            except Exception:
                # Fallback to AutoPipeline
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    cache_dir=str(CACHE_DIR),
                )
        
        report_progress(0.6, "Moving to GPU...")
        pipeline = pipeline.to("cuda")
        
        report_progress(0.8, "Applying optimizations...")
        # Memory optimizations
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            if hasattr(pipeline.vae, 'enable_slicing'):
                pipeline.vae.enable_slicing()
            if hasattr(pipeline.vae, 'enable_tiling'):
                pipeline.vae.enable_tiling()
        
        # Disable safety checker if present
        if hasattr(pipeline, 'safety_checker'):
            pipeline.safety_checker = None
        
        _current_model = pipeline
        _current_model_key = model_key
        _loaded_models[model_key] = pipeline
        
        report_progress(1.0, f"{model_key} loaded successfully!")
        logger.info(f"Model loaded: {model_key}")
        
        return True, f"{model_key} loaded successfully"
        
    except Exception as e:
        logger.error(f"Error loading model {model_key}: {e}")
        return False, f"Error loading {model_key}: {str(e)}"


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    guidance_scale: float = 7.0,
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    save_to_disk: bool = True,
) -> Optional[GenerationResult]:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt: The text prompt
        negative_prompt: Things to avoid in the image
        steps: Number of inference steps
        guidance_scale: How closely to follow the prompt
        width: Output image width
        height: Output image height
        seed: Random seed (-1 for random)
        lora_name: Name of LoRA to apply (None or "None" to skip)
        lora_strength: LoRA strength (0.0 to 1.5)
        progress_callback: Optional callback(current_step, total_steps)
        cancel_callback: Optional callback() -> bool, returns True to cancel
        save_to_disk: Whether to save the image to disk
    
    Returns:
        GenerationResult or None if generation fails or cancelled
    """
    global _current_model, _current_model_key, _current_lora
    
    if _current_model is None:
        logger.error("No model loaded")
        return None
    
    if not prompt.strip():
        logger.error("Empty prompt")
        return None
    
    # Get model info for type-specific handling
    models = get_available_image_models()
    model_info = models.get(_current_model_key, {})
    model_type = model_info.get("type", "sd")
    
    # Set up seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Progress callback wrapper with cancellation support
    class GenerationCancelled(Exception):
        pass
    
    def step_callback(pipe, step: int, timestep: int, callback_kwargs):
        # Check for cancellation
        if cancel_callback and cancel_callback():
            raise GenerationCancelled("Generation cancelled by user")
        if progress_callback:
            progress_callback(step, steps)
        return callback_kwargs
    
    logger.info(f"Generating: '{prompt[:50]}...' with {_current_model_key}")
    start_time = time.time()
    
    # Handle LoRA loading/unloading
    lora_loaded = False
    if lora_name and lora_name != "None":
        # Find the LoRA
        lora_path = LORA_DIR / lora_name
        if lora_path.exists():
            try:
                # Unload previous LoRA if different
                if _current_lora and _current_lora != lora_name:
                    unload_lora()
                
                if _current_lora != lora_name:
                    logger.info(f"Loading LoRA: {lora_name} with strength {lora_strength}")
                    _current_model.load_lora_weights(str(lora_path))
                    _current_model.fuse_lora(lora_scale=lora_strength)
                    _current_lora = lora_name
                    lora_loaded = True
                    logger.info(f"LoRA loaded: {lora_name}")
                else:
                    # Same LoRA, just update strength by re-fusing
                    _current_model.fuse_lora(lora_scale=lora_strength)
                    lora_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load LoRA {lora_name}: {e}")
        else:
            logger.warning(f"LoRA not found: {lora_path}")
    elif _current_lora:
        # No LoRA requested but one is loaded - unload it
        unload_lora()
    
    try:
        # Type-specific generation
        if model_type == "flux":
            result = _current_model(
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                generator=generator,
                callback_on_step_end=step_callback,
            )
        elif model_type == "sdxl-lightning":
            # SDXL Lightning requires guidance_scale=0.0 for best results
            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": 0.0,  # Lightning models need CFG=0
                "width": width,
                "height": height,
                "generator": generator,
                "callback_on_step_end": step_callback,
            }
            if negative_prompt.strip():
                gen_kwargs["negative_prompt"] = negative_prompt
            
            result = _current_model(**gen_kwargs)
        else:
            # SD / SDXL / SD3
            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "generator": generator,
                "callback_on_step_end": step_callback,
            }
            if negative_prompt.strip():
                gen_kwargs["negative_prompt"] = negative_prompt
            
            result = _current_model(**gen_kwargs)
        
        generation_time = time.time() - start_time
        image = result.images[0]
        
        # Save to disk
        filepath = None
        if save_to_disk:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = str(GENERATED_IMAGES_DIR / f"img_{timestamp}_{seed}.png")
            
            # Add metadata
            from PIL import PngImagePlugin
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("negative_prompt", negative_prompt)
            metadata.add_text("steps", str(steps))
            metadata.add_text("guidance_scale", str(guidance_scale))
            metadata.add_text("seed", str(seed))
            metadata.add_text("model", _current_model_key or "unknown")
            if lora_name and lora_name != "None":
                metadata.add_text("lora", lora_name)
                metadata.add_text("lora_strength", str(lora_strength))
            
            image.save(filepath, pnginfo=metadata)
            logger.info(f"Saved: {filepath}")
        
        # Convert to base64 for UI display
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode()
        
        return GenerationResult(
            image=image,
            seed=seed,
            generation_time=generation_time,
            filepath=filepath,
            base64_data=base64_data,
        )
    
    except GenerationCancelled:
        logger.info("Generation cancelled by user")
        return None
        
    except Exception as e:
        import traceback
        logger.error(f"Generation error: {e}")
        logger.error(traceback.format_exc())
        return None


def get_generated_images(limit: int = 50) -> List[Dict[str, Any]]:
    """Get list of recently generated images."""
    images = []
    
    if not GENERATED_IMAGES_DIR.exists():
        return images
    
    # Get PNG files sorted by modification time (newest first)
    files = sorted(
        GENERATED_IMAGES_DIR.glob("*.png"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )[:limit]
    
    for f in files:
        try:
            # Read metadata from PNG
            img = Image.open(f)
            metadata = img.info
            
            images.append({
                "path": str(f),
                "filename": f.name,
                "prompt": metadata.get("prompt", ""),
                "model": metadata.get("model", "unknown"),
                "seed": metadata.get("seed", ""),
                "timestamp": time.ctime(f.stat().st_mtime),
            })
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
    
    return images


def download_model(
    repo_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Download a HuggingFace model.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "Lykon/dreamshaper-8")
        progress_callback: Optional callback(progress: 0-1, message: str)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    from huggingface_hub import snapshot_download
    import threading
    
    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct*100:.0f}%] {msg}")
    
    # Check if already downloaded
    if is_model_downloaded(repo_id):
        return True, f"{repo_id} is already downloaded"
    
    report_progress(0.05, f"Starting download: {repo_id}")
    
    try:
        # Disable xet storage for more reliable downloads
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        
        report_progress(0.1, "Connecting to HuggingFace...")
        
        path = snapshot_download(
            repo_id,
            cache_dir=str(CACHE_DIR),
            max_workers=1,  # Single-threaded to avoid timeouts
            resume_download=True,
        )
        
        report_progress(1.0, f"Download complete: {repo_id}")
        logger.info(f"Model downloaded to: {path}")
        return True, f"Successfully downloaded {repo_id}"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Download error: {error_msg}")
        return False, f"Download failed: {error_msg}"
