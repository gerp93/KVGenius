"""
Core image generation engine for KVGenius.
UI-agnostic - talks to a local ComfyUI server, does not load diffusion
pipelines in-process.
"""
import os
import time
import random
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass

from PIL import Image

from . import comfyui_client
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
_current_model: Optional[Dict[str, Any]] = None  # {"family", "ckpt_name", "model_type"}
_current_model_key: Optional[str] = None
_current_lora: Optional[str] = None  # Track selected LoRA (informational; applied per-request)


@dataclass
class GenerationResult:
    """Result of image generation."""
    image: Image.Image
    seed: int
    generation_time: float
    filepath: Optional[str] = None
    base64_data: Optional[str] = None


def _family_for_type(model_type: str) -> str:
    """Map a preset's model type to a ComfyUI workflow template family."""
    if model_type in ("sdxl", "sdxl-lightning", "sd3"):
        return "sdxl"
    if model_type == "flux":
        return "flux"
    return "sd15"


def get_available_image_models() -> Dict[str, Dict[str, Any]]:
    """Get all available image models (HuggingFace + Local).

    'ckpt_filename' is the filename ComfyUI's CheckpointLoaderSimple expects
    (the checkpoint must already exist in ComfyUI's own checkpoints folder,
    or a folder ComfyUI is configured to see via extra_model_paths.yaml).
    'id' is kept as the HuggingFace repo id for the download/is-downloaded flow.
    """
    models = {}

    # Add HuggingFace models
    for name, preset in HUGGINGFACE_MODELS.items():
        models[name] = {
            "id": preset.get("id", ""),
            "ckpt_filename": preset.get("ckpt_filename", ""),
            "type": preset.get("type", "sd"),
            "source": "huggingface",
            "description": preset.get("description", ""),
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
                    "ckpt_filename": f.name,
                    "type": "sdxl" if is_sdxl else "sd",
                    "source": "local",
                    "description": f"Local checkpoint ({size_gb:.1f} GB)",
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


def _is_model_fully_downloaded(repo_id: str) -> bool:
    """Check if a HuggingFace model is fully downloaded.

    Checks:
    1. Cache folder exists with refs/ and snapshots/
    2. No .incomplete files in blobs/ (HF writes these during active downloads)
    3. Has substantial content (>100MB)
    """
    cache_name = "models--" + repo_id.replace("/", "--")
    cache_path = CACHE_DIR / cache_name

    if not cache_path.exists():
        return False

    # Check for .incomplete files in blobs/ - these indicate an in-progress or interrupted download
    blobs_path = cache_path / "blobs"
    if blobs_path.exists():
        incomplete_files = list(blobs_path.glob("*.incomplete"))
        if incomplete_files:
            return False  # Still has incomplete downloads

    # Verify refs/ and snapshots/ exist (created by snapshot_download)
    refs_path = cache_path / "refs"
    if not refs_path.exists():
        return False

    ref_files = list(refs_path.iterdir()) if refs_path.is_dir() else []
    if not ref_files:
        return False

    try:
        commit_hash = ref_files[0].read_text().strip()
        snapshot_path = cache_path / "snapshots" / commit_hash
        if not snapshot_path.exists():
            return False
        snapshot_files = list(snapshot_path.rglob('*'))
        if len(snapshot_files) < 2:
            return False
    except Exception:
        return False

    # Check minimum size to confirm files aren't just stubs
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
    """Get the currently loaded model info and its key."""
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
    """Clear the tracked LoRA selection (ComfyUI applies LoRA per-request, nothing to unload server-side)."""
    global _current_lora
    _current_lora = None


def unload_image_model():
    """Clear the tracked image model (ComfyUI manages its own VRAM/model lifecycle)."""
    global _current_model, _current_model_key
    if _current_model is not None:
        logger.info(f"Unloading image model: {_current_model_key}")
    _current_model = None
    _current_model_key = None


def load_image_model(
    model_key: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Point KVGenius at a checkpoint ComfyUI should use for generation.
    Does not load weights itself - ComfyUI manages that server-side; this
    validates the checkpoint is visible to ComfyUI.

    Args:
        model_key: Name of the model preset to load
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Tuple of (success: bool, message: str)
    """
    global _current_model, _current_model_key

    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct*100:.0f}%] {msg}")

    models = get_available_image_models()
    if model_key not in models:
        return False, f"Unknown model: {model_key}"

    model_info = models[model_key]
    ckpt_name = model_info.get("ckpt_filename") or model_info.get("id", "")
    model_type = model_info.get("type", "sd")

    if not ckpt_name:
        return False, f"{model_key} has no ComfyUI checkpoint filename configured"

    if _current_model_key == model_key and _current_model is not None:
        return True, f"{model_key} already loaded"

    report_progress(0.2, f"Checking ComfyUI for {ckpt_name}...")

    try:
        available = comfyui_client.list_checkpoints()
    except comfyui_client.ComfyUIUnavailableError as e:
        return False, str(e)

    if available and ckpt_name not in available:
        preview = ", ".join(available[:5]) + ("..." if len(available) > 5 else "")
        return False, f"'{ckpt_name}' not found in ComfyUI's checkpoints folder. Available: {preview}"

    _current_model = {
        "family": _family_for_type(model_type),
        "ckpt_name": ckpt_name,
        "model_type": model_type,
    }
    _current_model_key = model_key

    report_progress(1.0, f"{model_key} ready!")
    logger.info(f"Model ready: {model_key}")

    return True, f"{model_key} ready"


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
    Generate an image from a text prompt via ComfyUI.

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
        progress_callback: Optional callback(current_step, total_steps) - coarse, see comfyui_client.generate
        cancel_callback: Optional callback() -> bool, returns True to cancel
        save_to_disk: Whether to save the image to disk

    Returns:
        GenerationResult or None if generation fails or is cancelled
    """
    global _current_model, _current_model_key, _current_lora

    if _current_model is None:
        logger.error("No model loaded")
        return None

    if not prompt.strip():
        logger.error("Empty prompt")
        return None

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # SDXL Lightning models need CFG=0 for best results, regardless of caller-passed guidance
    cfg = 0.0 if _current_model.get("model_type") == "sdxl-lightning" else guidance_scale

    lora_file = lora_name if (lora_name and lora_name != "None") else None
    _current_lora = lora_file

    logger.info(f"Generating: '{prompt[:50]}...' with {_current_model_key}")
    start_time = time.time()

    try:
        image_bytes = comfyui_client.generate(
            family=_current_model["family"],
            ckpt_name=_current_model["ckpt_name"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            seed=seed,
            lora_name=lora_file,
            lora_strength=lora_strength,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    except comfyui_client.GenerationCancelled:
        logger.info("Generation cancelled by user")
        return None
    except comfyui_client.ComfyUIUnavailableError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None

    generation_time = time.time() - start_time

    image = Image.open(BytesIO(image_bytes))
    image.load()

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
        metadata.add_text("guidance_scale", str(cfg))
        metadata.add_text("seed", str(seed))
        metadata.add_text("model", _current_model_key or "unknown")
        if lora_file:
            metadata.add_text("lora", lora_file)
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
    Download a HuggingFace checkpoint into KVGenius's local checkpoints folder.

    Note: this only downloads the file locally - ComfyUI must also be able to
    see it (either point ComfyUI's extra_model_paths.yaml at this repo's
    data/checkpoints directory, or copy/symlink the file into ComfyUI's own
    checkpoints folder) before it can be selected for generation.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Lykon/dreamshaper-8")
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Tuple of (success: bool, message: str)
    """
    from huggingface_hub import snapshot_download, HfApi
    import threading

    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct*100:.0f}%] {msg}")

    # Check if already fully downloaded
    if _is_model_fully_downloaded(repo_id):
        return True, f"{repo_id} is already downloaded"

    report_progress(0.02, f"Starting download: {repo_id}")

    stop_monitor = threading.Event()

    try:
        # Disable xet storage for more reliable downloads
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        report_progress(0.05, "Fetching model info from HuggingFace...")

        # Get total size for progress tracking - try multiple methods
        cache_name = "models--" + repo_id.replace("/", "--")
        cache_path = CACHE_DIR / cache_name
        total_size = 0

        try:
            api = HfApi()
            # Method 1: model_info with files_metadata
            info = api.model_info(repo_id, files_metadata=True)
            siblings = info.siblings or []
            total_size = sum(s.size or 0 for s in siblings)

            # Method 2: if sizes were all 0, try list_repo_tree
            if total_size == 0 and siblings:
                try:
                    tree = list(api.list_repo_tree(repo_id, recursive=True))
                    total_size = sum(getattr(item, 'size', 0) or 0 for item in tree)
                except Exception:
                    pass

            file_count = len(siblings)
            if total_size > 0:
                report_progress(0.08, f"Found {file_count} files ({total_size / (1024**3):.1f} GB total)")
            else:
                report_progress(0.08, f"Found {file_count} files (calculating size...)")
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            file_count = 0

        # Track progress via a monitoring thread - ALWAYS start it
        last_reported_size = [0]  # Use list for mutability in closure

        def monitor_progress():
            """Periodically check downloaded size and report progress."""
            while not stop_monitor.is_set():
                try:
                    if cache_path.exists():
                        # Only count blobs/ directory to avoid double-counting symlinked snapshots
                        blobs_path = cache_path / "blobs"
                        if blobs_path.exists():
                            downloaded = sum(
                                f.stat().st_size for f in blobs_path.iterdir() if f.is_file()
                            )
                        else:
                            downloaded = sum(
                                f.stat().st_size for f in cache_path.rglob('*') if f.is_file()
                            )

                        # Only report if size actually changed
                        if downloaded != last_reported_size[0]:
                            last_reported_size[0] = downloaded
                            size_gb = downloaded / (1024**3)

                            if total_size > 0:
                                pct = min(downloaded / total_size, 0.98)
                                total_gb = total_size / (1024**3)
                                report_progress(pct, f"Downloading: {size_gb:.2f} / {total_gb:.2f} GB")
                            else:
                                # Unknown total - show size downloaded with indeterminate progress
                                report_progress(min(0.5, size_gb / 20.0), f"Downloading: {size_gb:.2f} GB...")
                except Exception:
                    pass
                stop_monitor.wait(3.0)  # Check every 3 seconds

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        path = snapshot_download(
            repo_id,
            cache_dir=str(CACHE_DIR),
            max_workers=1,  # Single-threaded to avoid timeouts
            resume_download=True,
        )

        stop_monitor.set()

        report_progress(1.0, f"Download complete: {repo_id}")
        logger.info(f"Model downloaded to: {path}")
        return True, f"Successfully downloaded {repo_id}"

    except Exception as e:
        stop_monitor.set()
        error_msg = str(e)
        logger.error(f"Download error: {error_msg}")
        return False, f"Download failed: {error_msg}"
