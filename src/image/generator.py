"""
Image generation using Stable Diffusion models.
"""
import os
import time
import logging
import torch
from PIL import PngImagePlugin

logger = logging.getLogger(__name__)


def read_image_metadata(filepath):
    """Read generation metadata from a PNG file.
    
    Args:
        filepath: Path to the PNG file
    
    Returns:
        Dict of metadata or empty dict if not found
    """
    try:
        from PIL import Image
        img = Image.open(filepath)
        metadata = {}
        if hasattr(img, 'info'):
            for key in ['prompt', 'negative_prompt', 'steps', 'guidance_scale', 
                       'width', 'height', 'seed', 'model', 'lora', 'lora_strength', 'generated_at']:
                if key in img.info:
                    metadata[key] = img.info[key]
        return metadata
    except Exception as e:
        logger.error(f"Error reading metadata from {filepath}: {e}")
        return {}


def format_image_metadata(metadata):
    """Format metadata for display.
    
    Args:
        metadata: Dict of metadata
    
    Returns:
        Formatted markdown string
    """
    if not metadata:
        return "*No metadata available*"
    
    lines = []
    if metadata.get('prompt'):
        lines.append(f"**Prompt:** {metadata['prompt']}")
    if metadata.get('negative_prompt'):
        lines.append(f"**Negative:** {metadata['negative_prompt']}")
    if metadata.get('model'):
        lines.append(f"**Model:** {metadata['model']}")
    
    settings = []
    if metadata.get('steps'):
        settings.append(f"Steps: {metadata['steps']}")
    if metadata.get('guidance_scale'):
        settings.append(f"CFG: {metadata['guidance_scale']}")
    if metadata.get('width') and metadata.get('height'):
        settings.append(f"Size: {metadata['width']}x{metadata['height']}")
    if metadata.get('seed'):
        settings.append(f"Seed: {metadata['seed']}")
    
    if settings:
        lines.append(f"**Settings:** {' | '.join(settings)}")
    
    if metadata.get('lora') and metadata['lora'] != "None":
        lora_info = metadata['lora']
        if metadata.get('lora_strength'):
            lora_info += f" ({metadata['lora_strength']})"
        lines.append(f"**LoRA:** {lora_info}")
    
    if metadata.get('generated_at'):
        lines.append(f"**Generated:** {metadata['generated_at']}")
    
    return "\n\n".join(lines)


class ImageGenerator:
    """Handles Stable Diffusion image generation."""
    
    def __init__(self, cache_dir="./data/model_cache", output_dir="./data/generated_images", 
                 lora_dir="./data/lora_models"):
        """Initialize the image generator.
        
        Args:
            cache_dir: Directory for caching models
            output_dir: Directory for saving generated images
            lora_dir: Directory for LoRA models
        """
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.lora_dir = lora_dir
        
        self.model = None
        self.loaded_models = {}  # Cache of loaded pipelines by name
        self.current_model_key = None
        self.model_loaded = False
        
        # Generation state
        self.cancel_requested = False
        self.in_progress = False
    
    def is_model_downloaded(self, repo_id):
        """Check if a model is downloaded to disk.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., 'Lykon/dreamshaper-8')
        
        Returns:
            True if model is fully downloaded
        """
        # HuggingFace cache structure: models--org--name
        cache_name = "models--" + repo_id.replace("/", "--")
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        if not os.path.exists(cache_path):
            return False
        
        # Check for snapshots folder with actual model files
        snapshots_path = os.path.join(cache_path, "snapshots")
        if os.path.exists(snapshots_path):
            for snapshot in os.listdir(snapshots_path):
                snapshot_dir = os.path.join(snapshots_path, snapshot)
                if os.path.isdir(snapshot_dir):
                    # For SD models, check that unet is fully downloaded
                    unet_path = os.path.join(snapshot_dir, "unet")
                    if os.path.isdir(unet_path):
                        unet_files = [f for f in os.listdir(unet_path) 
                                      if f.endswith(('.safetensors', '.bin'))]
                        if unet_files:
                            # Check file isn't being downloaded (has .incomplete suffix or is very small)
                            for uf in unet_files:
                                full_path = os.path.join(unet_path, uf)
                                # UNet should be at least 1GB for SD models
                                if os.path.getsize(full_path) > 1_000_000_000:
                                    return True
        return False
    
    def unload_all(self):
        """Unload all cached image models to free VRAM."""
        for model_key in list(self.loaded_models.keys()):
            try:
                del self.loaded_models[model_key]
                logger.info(f"Unloaded image model: {model_key}")
            except:
                pass
        
        self.loaded_models.clear()
        self.model = None
        self.model_loaded = False
        self.current_model_key = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_from_checkpoint(self, model_key, checkpoint_path, progress_callback=None):
        """Load a Stable Diffusion model from a single checkpoint file.
        
        Args:
            model_key: Display name of the model
            checkpoint_path: Path to .safetensors or .ckpt file
            progress_callback: Optional callback for progress updates
        
        Returns:
            Status message string
        """
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        
        if progress_callback:
            progress_callback(0.4, "Loading checkpoint...")
        
        # Detect if SDXL based on file size (SDXL is ~6.5GB, SD1.5 is ~2GB)
        file_size_gb = os.path.getsize(checkpoint_path) / (1024**3)
        is_sdxl = file_size_gb > 4.0 or "xl" in checkpoint_path.lower() or "sdxl" in checkpoint_path.lower()
        
        logger.info(f"Loading checkpoint: {checkpoint_path} (size: {file_size_gb:.1f}GB, is_sdxl: {is_sdxl})")
        
        try:
            if is_sdxl:
                self.model = StableDiffusionXLPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith('.safetensors'),
                )
            else:
                self.model = StableDiffusionPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith('.safetensors'),
                )
        except Exception as e:
            logger.error(f"Failed to load checkpoint with auto-detect: {e}")
            # Try the other pipeline type
            try:
                if is_sdxl:
                    self.model = StableDiffusionPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=torch.float16,
                    )
                else:
                    self.model = StableDiffusionXLPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=torch.float16,
                    )
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint: {e}, {e2}")
        
        if progress_callback:
            progress_callback(0.8, "Moving to GPU...")
        self.model = self.model.to("cuda")
        
        # Disable NSFW safety checker
        if hasattr(self.model, 'safety_checker'):
            self.model.safety_checker = None
        if hasattr(self.model, 'requires_safety_checker'):
            self.model.requires_safety_checker = False
        
        # Enable memory optimizations
        self.model.enable_attention_slicing()
        
        if progress_callback:
            progress_callback(1.0, "Done!")
        
        # Track loaded model
        self.loaded_models[model_key] = self.model
        self.current_model_key = model_key
        self.model_loaded = True
        
        checkpoint_name = os.path.basename(checkpoint_path)
        logger.info(f"Checkpoint loaded: {model_key} from {checkpoint_name}")
        return f"✅ **{model_key}** loaded from checkpoint!"

    def load_model(self, model_key, model_id, progress_callback=None):
        """Load a Stable Diffusion model.
        
        Args:
            model_key: Display name of the model
            model_id: HuggingFace model ID OR local path to checkpoint file
            progress_callback: Optional callback for progress updates
        
        Returns:
            Status message string
        """
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
            
            if progress_callback:
                progress_callback(0.1, "Preparing...")
            logger.info(f"Loading image model: {model_id}")
            
            # Auto-unload any previously loaded model to free VRAM
            if self.loaded_models:
                prev_models = list(self.loaded_models.keys())
                if progress_callback:
                    progress_callback(0.15, "Unloading previous model...")
                self.unload_all()
                logger.info(f"Auto-unloaded previous models: {prev_models}")
            
            if progress_callback:
                progress_callback(0.3, "Downloading/loading pipeline...")
            
            # Check if this is a local checkpoint file
            is_checkpoint = False
            checkpoint_path = None
            
            # Check if model_id is a local file path
            if os.path.isfile(model_id):
                is_checkpoint = True
                checkpoint_path = model_id
            else:
                # Check in checkpoints directory
                checkpoints_dir = os.path.join(os.path.dirname(self.cache_dir), "checkpoints")
                potential_path = os.path.join(checkpoints_dir, model_id)
                if os.path.isfile(potential_path):
                    is_checkpoint = True
                    checkpoint_path = potential_path
                # Also check if model_id is just a filename without path
                for ext in ['.safetensors', '.ckpt']:
                    potential_path = os.path.join(checkpoints_dir, f"{model_id}{ext}")
                    if os.path.isfile(potential_path):
                        is_checkpoint = True
                        checkpoint_path = potential_path
                        break
            
            if is_checkpoint:
                logger.info(f"Loading checkpoint file: {checkpoint_path}")
                return self._load_from_checkpoint(model_key, checkpoint_path, progress_callback)
            
            # Load appropriate pipeline based on model (HuggingFace format)
            # Detect SDXL by name hints or auto-detect from model config
            is_sdxl = "sdxl" in model_id.lower() or "xl" in model_id.lower() or "turbo" in model_key.lower()
            
            # Also check if model has SDXL structure from model_index.json
            if not is_sdxl:
                try:
                    from huggingface_hub import hf_hub_download
                    import json
                    config_path = hf_hub_download(model_id, "model_index.json", cache_dir=self.cache_dir)
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if config.get('_class_name') == 'StableDiffusionXLPipeline' or 'text_encoder_2' in config:
                        is_sdxl = True
                        logger.info(f"Auto-detected SDXL model from config: {model_id}")
                except:
                    pass  # Fall back to name-based detection
            
            logger.info(f"Loading model: {model_id}, is_sdxl={is_sdxl}")
            
            if is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                # Try without variant first (most SDXL models don't have fp16 variant)
                for variant in [None, "fp16"]:
                    try:
                        logger.info(f"Trying SDXL pipeline with variant={variant}")
                        self.model = StableDiffusionXLPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16,
                            cache_dir=self.cache_dir,
                            variant=variant,
                            use_safetensors=True,
                        )
                        logger.info(f"SDXL pipeline loaded successfully with variant={variant}")
                        break
                    except Exception as e:
                        logger.warning(f"SDXL load with variant={variant} failed: {e}")
                        continue
                else:
                    # All SDXL attempts failed, try AutoPipeline as last resort
                    logger.warning("All SDXL variants failed, trying AutoPipeline")
                    self.model = AutoPipelineForText2Image.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        cache_dir=self.cache_dir,
                    )
            else:
                # Try loading with different variants - some models only have fp16 safetensors
                load_errors = []
                for variant, use_st in [("fp16", True), (None, True), (None, False)]:
                    try:
                        kwargs = {
                            "torch_dtype": torch.float16,
                            "cache_dir": self.cache_dir,
                        }
                        if variant:
                            kwargs["variant"] = variant
                        if use_st:
                            kwargs["use_safetensors"] = True
                        
                        self.model = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
                        logger.info(f"Loaded model with variant={variant}, safetensors={use_st}")
                        break
                    except Exception as e:
                        load_errors.append(f"variant={variant}, safetensors={use_st}: {e}")
                        continue
                else:
                    # All attempts failed
                    raise RuntimeError(f"Failed to load model. Tried:\n" + "\n".join(load_errors))
            
            if progress_callback:
                progress_callback(0.8, "Moving to GPU...")
            self.model = self.model.to("cuda")
            
            # Disable NSFW safety checker (causes black images on false positives)
            if hasattr(self.model, 'safety_checker'):
                self.model.safety_checker = None
            if hasattr(self.model, 'requires_safety_checker'):
                self.model.requires_safety_checker = False
            
            # Enable memory optimizations
            self.model.enable_attention_slicing()
            
            if progress_callback:
                progress_callback(1.0, "Done!")
            
            # Track loaded model
            self.loaded_models[model_key] = self.model
            self.current_model_key = model_key
            self.model_loaded = True
            logger.info(f"Image model loaded: {model_key}")

            return f"✅ **{model_key}** loaded successfully!"
            
        except ImportError:
            return "❌ Please install diffusers: `pip install diffusers transformers accelerate`"
        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            return f"❌ Error loading model: {str(e)}"
    
    def generate(self, prompt, negative_prompt="", num_steps=25, guidance_scale=7.5,
                 width=512, height=512, seed=-1, lora_name=None, lora_strength=0.8,
                 progress_callback=None):
        """Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (things to avoid)
            num_steps: Number of inference steps
            guidance_scale: CFG scale
            width: Image width
            height: Image height
            seed: Random seed (-1 for random)
            lora_name: Optional LoRA name to apply
            lora_strength: LoRA influence strength
            progress_callback: Optional callback for progress updates
        
        Returns:
            Tuple of (PIL.Image or None, status_message)
        """
        if self.model is None:
            return None, "❌ No image model loaded. Please load a model first."
        
        if not prompt.strip():
            return None, "❌ Please enter a prompt."
        
        try:
            if progress_callback:
                progress_callback(0.1, "Preparing generation...")
            
            # Load LoRA if specified
            lora_loaded = False
            loaded_lora_names = []
            if lora_name and lora_name != "None":
                # Support single LoRA or list of LoRAs
                lora_names = lora_name if isinstance(lora_name, list) else [lora_name]
                lora_weights = lora_strength if isinstance(lora_strength, list) else [lora_strength] * len(lora_names)
                
                for idx, (ln, lw) in enumerate(zip(lora_names, lora_weights)):
                    try:
                        lora_path = os.path.join(self.lora_dir, ln)
                        if os.path.exists(lora_path):
                            if progress_callback:
                                progress_callback(0.15 + idx * 0.02, f"Loading LoRA: {ln}...")
                            print(f"[LoRA] Loading from: {lora_path}")
                            
                            # Check for PEFT format (adapter_config.json + adapter_model.safetensors)
                            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
                            adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
                            
                            if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
                                # PEFT format - load with adapter_name for multi-LoRA support
                                adapter_name = f"lora_{idx}"
                                self.model.load_lora_weights(lora_path, adapter_name=adapter_name)
                                loaded_lora_names.append((adapter_name, lw))
                                print(f"[LoRA] ✓ Loaded PEFT adapter '{ln}' as '{adapter_name}'")
                            else:
                                # Look for .safetensors file directly
                                safetensor_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
                                if safetensor_files:
                                    safetensor_path = os.path.join(lora_path, safetensor_files[0])
                                    adapter_name = f"lora_{idx}"
                                    self.model.load_lora_weights(safetensor_path, adapter_name=adapter_name)
                                    loaded_lora_names.append((adapter_name, lw))
                                    print(f"[LoRA] ✓ Loaded safetensors '{ln}' as '{adapter_name}'")
                                else:
                                    print(f"[LoRA] ✗ No valid LoRA files found in: {lora_path}")
                                    continue
                            
                            lora_loaded = True
                            logger.info(f"Loaded LoRA: {ln} with strength {lw}")
                        else:
                            print(f"[LoRA] ✗ Path not found: {lora_path}")
                    except Exception as e:
                        print(f"[LoRA] ✗ Failed to load {ln}: {e}")
                        logger.warning(f"Failed to load LoRA {ln}: {e}")
                
                # Set adapter weights if multiple LoRAs loaded
                if len(loaded_lora_names) > 0:
                    try:
                        adapter_names = [name for name, _ in loaded_lora_names]
                        adapter_weights = [weight for _, weight in loaded_lora_names]
                        self.model.set_adapters(adapter_names, adapter_weights=adapter_weights)
                        print(f"[LoRA] Set adapters: {list(zip(adapter_names, adapter_weights))}")
                    except Exception as e:
                        # Fall back to fuse_lora for single LoRA if set_adapters not supported
                        if len(loaded_lora_names) == 1:
                            try:
                                self.model.fuse_lora(lora_scale=loaded_lora_names[0][1])
                                print(f"[LoRA] Fused single LoRA with scale {loaded_lora_names[0][1]}")
                            except Exception as e2:
                                print(f"[LoRA] Warning: Could not set weights: {e2}")
            
            # Set seed for reproducibility
            generator = None
            if seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(seed)
            
            gen_start = time.time()
            
            if progress_callback:
                progress_callback(0.2, "Generating image...")

            # Setup cancel flag and callback
            self.cancel_requested = False
            self.in_progress = True

            def _generation_callback(pipe, step, timestep, callback_kwargs):
                if self.cancel_requested:
                    raise Exception("Generation cancelled by user")
                return callback_kwargs

            # Prepare negative prompt (handle None case)
            neg_prompt = None
            if negative_prompt and isinstance(negative_prompt, str) and negative_prompt.strip():
                neg_prompt = negative_prompt

            # Generate image - use callback_on_step_end for newer diffusers
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "generator": generator,
            }
            
            # Try new callback API first, fall back to legacy
            try:
                result = self.model(
                    **gen_kwargs,
                    callback_on_step_end=_generation_callback,
                )
            except TypeError:
                # Legacy callback for older models/pipelines
                def _legacy_callback(step, timestep, latents):
                    if self.cancel_requested:
                        raise Exception("Generation cancelled by user")
                result = self.model(
                    **gen_kwargs,
                    callback=_legacy_callback,
                    callback_steps=1,
                )
            
            gen_time = time.time() - gen_start
            
            # Unload LoRA after generation to keep model clean
            if lora_loaded:
                try:
                    # Disable adapters first (for multi-LoRA)
                    try:
                        self.model.disable_lora()
                    except:
                        pass
                    # Then try unfuse (for fused single LoRA)
                    try:
                        self.model.unfuse_lora()
                    except:
                        pass
                    # Finally unload weights
                    self.model.unload_lora_weights()
                except Exception as e:
                    logger.warning(f"Failed to unload LoRA: {e}")
            
            if progress_callback:
                progress_callback(1.0, "Done!")
            
            image = result.images[0]
            
            # Save to file with metadata
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"img_{timestamp}.png")
            
            # Add generation parameters as PNG metadata
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("negative_prompt", negative_prompt if negative_prompt else "")
            metadata.add_text("steps", str(num_steps))
            metadata.add_text("guidance_scale", str(guidance_scale))
            metadata.add_text("width", str(width))
            metadata.add_text("height", str(height))
            metadata.add_text("seed", str(seed))
            metadata.add_text("model", self.current_model_key if self.current_model_key else "unknown")
            metadata.add_text("lora", lora_name if lora_name and lora_name != "None" else "")
            metadata.add_text("lora_strength", str(lora_strength) if lora_name and lora_name != "None" else "")
            metadata.add_text("generated_at", timestamp)
            
            image.save(filename, pnginfo=metadata)
            
            status = f"✅ Generated in **{gen_time:.1f}s** | Saved to `{filename}`"
            if lora_name and lora_name != "None":
                status += f" | LoRA: {lora_name} ({lora_strength})"
            if seed != -1:
                status += f" | Seed: {seed}"
            
            self.in_progress = False
            return image, status
            
        except Exception as e:
            self.in_progress = False
            # Unload LoRA on error too
            if lora_name and lora_name != "None":
                try:
                    self.model.disable_lora()
                except:
                    pass
                try:
                    self.model.unfuse_lora()
                except:
                    pass
                try:
                    self.model.unload_lora_weights()
                except:
                    pass
            # If cancelled by user, return a friendly message
            if str(e).lower().find('cancel') != -1:
                logger.info("Image generation cancelled by user")
                return None, "🛑 Generation cancelled"
            logger.error(f"Error generating image: {e}")
            return None, f"❌ Error: {str(e)}"
    
    def cancel_generation(self):
        """Request cancellation of current generation."""
        if self.in_progress:
            self.cancel_requested = True
            logger.info("Image generation cancel requested")
            return True
        return False
    
    def get_status(self):
        """Get current generator status.
        
        Returns:
            Status string for display
        """
        lines = []
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            lines.append(f"🎮 **VRAM:** {vram_used:.2f} / {vram_total:.1f} GB")
        
        if self.model_loaded:
            lines.append("✅ **Active image model:** Loaded")
        if self.loaded_models:
            lines.append(f"\n📦 **Cached models ({len(self.loaded_models)}):**")
            for name in self.loaded_models.keys():
                lines.append(f"  • {name}")
        else:
            lines.append("\n📦 **No cached image models**")
        
        if self.in_progress:
            lines.append("\n⏳ Image generation: In progress")
        else:
            lines.append("\n✅ Image generation: Idle")

        return "\n".join(lines)
    
    def get_generated_images(self):
        """Return list of generated image paths sorted by mtime desc.
        
        Returns:
            List of absolute file paths
        """
        if not os.path.isdir(self.output_dir):
            return []
        files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) 
                 if f.lower().endswith(".png")]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files
