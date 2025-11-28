"""
LoRA Training Module for KVGenius
Handles dataset management, training, and LoRA file management.
"""

import os
import json
import shutil
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, asdict
from PIL import Image

logger = logging.getLogger(__name__)

# Directories - all user data goes in data/
LORA_DIR = Path("./data/lora_models")
DATASETS_DIR = Path("./data/training_datasets")
MODEL_CACHE_DIR = Path("./data/model_cache")
LORA_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LoRAInfo:
    """Information about a trained LoRA."""
    name: str
    base_model: str
    trigger_word: str
    created_at: str
    training_steps: int
    rank: int
    alpha: float
    description: str = ""
    preview_image: Optional[str] = None
    file_path: Optional[str] = None
    file_size_mb: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""
    dataset_name: str
    output_name: str
    base_model: str
    trigger_word: str
    description: str = ""
    
    # Resume training from existing LoRA
    resume_from: Optional[str] = None  # Path to existing LoRA to continue training
    
    # Training params
    num_train_epochs: int = 100
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    
    # LoRA params
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Resolution
    resolution: int = 512
    center_crop: bool = True
    
    # Memory optimization
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    
    # Save
    save_steps: int = 500
    max_train_steps: Optional[int] = None


class DatasetManager:
    """Manages training datasets with images and captions."""
    
    def __init__(self, datasets_dir: Path = DATASETS_DIR):
        self.datasets_dir = datasets_dir
        self.datasets_dir.mkdir(exist_ok=True)
    
    def create_dataset(self, name: str) -> Path:
        """Create a new empty dataset folder."""
        dataset_path = self.datasets_dir / name
        dataset_path.mkdir(exist_ok=True)
        (dataset_path / "images").mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "images": []
        }
        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return dataset_path
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        datasets = []
        for item in self.datasets_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                datasets.append(item.name)
        return sorted(datasets)
    
    def get_dataset_info(self, name: str) -> Dict:
        """Get dataset metadata."""
        metadata_path = self.datasets_dir / name / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}
    
    def add_image(self, dataset_name: str, image: Image.Image, filename: str, caption: str = "") -> str:
        """Add an image to a dataset with optional caption."""
        dataset_path = self.datasets_dir / dataset_name
        images_dir = dataset_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Save image
        base_name = Path(filename).stem
        image_path = images_dir / f"{base_name}.png"
        
        # Handle duplicates
        counter = 1
        while image_path.exists():
            image_path = images_dir / f"{base_name}_{counter}.png"
            counter += 1
        
        image.save(image_path, "PNG")
        
        # Save caption as .txt file with same name
        caption_path = image_path.with_suffix(".txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)
        
        # Update metadata
        metadata = self.get_dataset_info(dataset_name)
        if "images" not in metadata:
            metadata["images"] = []
        
        metadata["images"].append({
            "filename": image_path.name,
            "caption": caption,
            "added_at": datetime.now().isoformat()
        })
        
        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(image_path)
    
    def get_images(self, dataset_name: str) -> List[Dict]:
        """Get all images in a dataset with their captions."""
        dataset_path = self.datasets_dir / dataset_name
        images_dir = dataset_path / "images"
        
        if not images_dir.exists():
            return []
        
        images = []
        for img_path in sorted(images_dir.glob("*.png")):
            caption_path = img_path.with_suffix(".txt")
            caption = ""
            if caption_path.exists():
                with open(caption_path, encoding="utf-8") as f:
                    caption = f.read().strip()
            
            images.append({
                "path": str(img_path),
                "filename": img_path.name,
                "caption": caption
            })
        
        # Also check for jpg/jpeg
        for ext in ["*.jpg", "*.jpeg"]:
            for img_path in sorted(images_dir.glob(ext)):
                caption_path = img_path.with_suffix(".txt")
                caption = ""
                if caption_path.exists():
                    with open(caption_path, encoding="utf-8") as f:
                        caption = f.read().strip()
                
                images.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "caption": caption
                })
        
        return images
    
    def update_caption(self, dataset_name: str, filename: str, new_caption: str):
        """Update the caption for an image."""
        dataset_path = self.datasets_dir / dataset_name / "images"
        
        # Find the image file
        image_path = dataset_path / filename
        if not image_path.exists():
            # Try without extension match
            for ext in [".png", ".jpg", ".jpeg"]:
                test_path = dataset_path / (Path(filename).stem + ext)
                if test_path.exists():
                    image_path = test_path
                    break
        
        if image_path.exists():
            caption_path = image_path.with_suffix(".txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(new_caption)
            return True
        return False
    
    def delete_image(self, dataset_name: str, filename: str) -> bool:
        """Delete an image and its caption from a dataset."""
        dataset_path = self.datasets_dir / dataset_name / "images"
        image_path = dataset_path / filename
        caption_path = image_path.with_suffix(".txt")
        
        deleted = False
        if image_path.exists():
            image_path.unlink()
            deleted = True
        if caption_path.exists():
            caption_path.unlink()
        
        return deleted
    
    def delete_dataset(self, name: str) -> bool:
        """Delete an entire dataset."""
        dataset_path = self.datasets_dir / name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False
    
    def crop_image(self, image: Image.Image, size: int = 512, 
                   crop_mode: str = "resize_pad") -> Image.Image:
        """
        Process image for training with different crop modes.
        
        Args:
            image: PIL Image to process
            size: Target size (default 512)
            crop_mode: One of:
                - "center": Center crop to square (may cut off edges)
                - "resize_pad": Resize to fit, pad with black (keeps everything)
                - "resize_stretch": Stretch to square (may distort)
                - "top": Crop from top (good for portraits)
                - "smart": Try to find subject and crop around it
        
        Returns:
            Processed PIL Image at target size
        """
        width, height = image.size
        
        if crop_mode == "resize_pad":
            # Resize to fit within square, pad the rest with black
            # This preserves ALL details - nothing is cut off
            ratio = min(size / width, size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create square canvas and paste centered
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            paste_x = (size - new_width) // 2
            paste_y = (size - new_height) // 2
            canvas.paste(resized, (paste_x, paste_y))
            return canvas
        
        elif crop_mode == "resize_stretch":
            # Just stretch to square - simple but may distort
            return image.resize((size, size), Image.Resampling.LANCZOS)
        
        elif crop_mode == "top":
            # Crop from top - good for portraits where head is at top
            if width > height:
                # Wide image - center horizontally, take full height
                left = (width - height) // 2
                image = image.crop((left, 0, left + height, height))
            else:
                # Tall image - take from top
                image = image.crop((0, 0, width, width))
            return image.resize((size, size), Image.Resampling.LANCZOS)
        
        elif crop_mode == "smart":
            # Try to detect subject and crop around it
            # For now, use center with some bias toward top (faces usually there)
            if width > height:
                left = (width - height) // 2
                image = image.crop((left, 0, left + height, height))
            else:
                # Bias toward top third for portraits
                crop_height = width
                top = max(0, (height - crop_height) // 3)  # Top third bias
                bottom = top + crop_height
                if bottom > height:
                    bottom = height
                    top = height - crop_height
                image = image.crop((0, top, width, bottom))
            return image.resize((size, size), Image.Resampling.LANCZOS)
        
        else:  # "center" - original behavior
            if width > height:
                left = (width - height) // 2
                right = left + height
                top, bottom = 0, height
            else:
                top = (height - width) // 2
                bottom = top + width
                left, right = 0, width
            
            image = image.crop((left, top, right, bottom))
            return image.resize((size, size), Image.Resampling.LANCZOS)


class LoRAManager:
    """Manages trained LoRA files."""
    
    def __init__(self, lora_dir: Path = LORA_DIR):
        self.lora_dir = Path(lora_dir) if isinstance(lora_dir, str) else lora_dir
        self.lora_dir.mkdir(exist_ok=True)
        self.metadata_file = self.lora_dir / "lora_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load LoRA metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"loras": {}}
    
    def _save_metadata(self):
        """Save LoRA metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def list_loras(self) -> List[LoRAInfo]:
        """List all available LoRAs."""
        loras = []
        
        # Check for LoRA subfolders (each LoRA saved in its own folder)
        for lora_folder in self.lora_dir.iterdir():
            if not lora_folder.is_dir():
                continue
            
            # Look for adapter_model.safetensors in the folder
            adapter_path = lora_folder / "adapter_model.safetensors"
            if not adapter_path.exists():
                continue
            
            name = lora_folder.name
            
            # Try to load metadata from the folder
            metadata_path = lora_folder / "lora_metadata.json"
            info = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        info = json.load(f)
                except:
                    pass
            
            # Also check the global metadata
            if not info:
                info = self.metadata.get("loras", {}).get(name, {})
            
            loras.append(LoRAInfo(
                name=name,
                base_model=info.get("base_model", "Unknown"),
                trigger_word=info.get("trigger_word", name),
                created_at=info.get("created_at", "Unknown"),
                training_steps=info.get("training_steps", 0),
                rank=info.get("rank", 16),
                alpha=info.get("alpha", 32),
                description=info.get("description", ""),
                preview_image=info.get("preview_image"),
                file_path=str(adapter_path),
                file_size_mb=adapter_path.stat().st_size / (1024 * 1024)
            ))
        
        return sorted(loras, key=lambda x: x.name)
    
    def get_lora(self, name: str) -> Optional[LoRAInfo]:
        """Get info for a specific LoRA."""
        lora_folder = self.lora_dir / name
        adapter_path = lora_folder / "adapter_model.safetensors"
        
        if not adapter_path.exists():
            return None
        
        # Try to load metadata from the folder
        metadata_path = lora_folder / "lora_metadata.json"
        info = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    info = json.load(f)
            except:
                pass
        
        # Also check the global metadata
        if not info:
            info = self.metadata.get("loras", {}).get(name, {})
        
        return LoRAInfo(
            name=name,
            base_model=info.get("base_model", "Unknown"),
            trigger_word=info.get("trigger_word", name),
            created_at=info.get("created_at", "Unknown"),
            training_steps=info.get("training_steps", 0),
            rank=info.get("rank", 16),
            alpha=info.get("alpha", 32),
            description=info.get("description", ""),
            preview_image=info.get("preview_image"),
            file_path=str(adapter_path),
            file_size_mb=adapter_path.stat().st_size / (1024 * 1024)
        )
    
    def register_lora(self, info: LoRAInfo):
        """Register a LoRA in metadata."""
        if "loras" not in self.metadata:
            self.metadata["loras"] = {}
        
        self.metadata["loras"][info.name] = {
            "base_model": info.base_model,
            "trigger_word": info.trigger_word,
            "created_at": info.created_at,
            "training_steps": info.training_steps,
            "rank": info.rank,
            "alpha": info.alpha,
            "description": info.description,
            "preview_image": info.preview_image
        }
        self._save_metadata()
    
    def delete_lora(self, name: str) -> bool:
        """Delete a LoRA and its metadata."""
        lora_path = self.lora_dir / f"{name}.safetensors"
        
        if lora_path.exists():
            lora_path.unlink()
        
        if name in self.metadata.get("loras", {}):
            del self.metadata["loras"][name]
            self._save_metadata()
        
        return True
    
    def import_lora(self, file_path: str, name: str, trigger_word: str, 
                    base_model: str = "Unknown", description: str = "") -> LoRAInfo:
        """Import an external LoRA file."""
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"LoRA file not found: {file_path}")
        
        dest = self.lora_dir / f"{name}.safetensors"
        shutil.copy2(source, dest)
        
        info = LoRAInfo(
            name=name,
            base_model=base_model,
            trigger_word=trigger_word,
            created_at=datetime.now().isoformat(),
            training_steps=0,
            rank=16,
            alpha=32,
            description=description,
            file_path=str(dest),
            file_size_mb=dest.stat().st_size / (1024 * 1024)
        )
        
        self.register_lora(info)
        return info


class TrainingState:
    """Holds the current training state for UI updates."""
    def __init__(self):
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss = 0.0
        self.status_message = "Idle"
        self.preview_images = []
        self.error = None
        self.completed = False
        self.output_path = None


# Global training state
training_state = TrainingState()


class LoRATrainer:
    """Handles LoRA training with background execution."""
    
    def __init__(self, cache_dir: str = "./data/model_cache"):
        self.cache_dir = cache_dir
        self.dataset_manager = DatasetManager()
        self.lora_manager = LoRAManager()
        self._training_thread: Optional[threading.Thread] = None
        self._stop_requested = False
    
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return training_state.is_training
    
    def get_state(self) -> TrainingState:
        """Get current training state."""
        return training_state
    
    def request_stop(self):
        """Request training to stop."""
        self._stop_requested = True
        training_state.status_message = "Stopping..."
    
    def start_training(self, config: TrainingConfig, 
                       progress_callback: Optional[Callable] = None) -> bool:
        """Start training in background thread."""
        global training_state
        
        if training_state.is_training:
            return False
        
        self._stop_requested = False
        training_state = TrainingState()
        training_state.is_training = True
        training_state.status_message = "Initializing..."
        
        def train_thread():
            try:
                self._run_training(config, progress_callback)
            except Exception as e:
                logger.error(f"Training error: {e}")
                training_state.error = str(e)
                training_state.status_message = f"Error: {e}"
            finally:
                training_state.is_training = False
        
        self._training_thread = threading.Thread(target=train_thread, daemon=True)
        self._training_thread.start()
        
        return True
    
    def _run_training(self, config: TrainingConfig, 
                      progress_callback: Optional[Callable] = None):
        """Run the actual training loop."""
        global training_state
        
        try:
            import torch
            from diffusers import StableDiffusionPipeline, DDPMScheduler
            from diffusers.loaders import LoraLoaderMixin
            from transformers import CLIPTextModel, CLIPTokenizer
            from peft import LoraConfig, get_peft_model
            from torch.utils.data import Dataset, DataLoader
            import torch.nn.functional as F
        except ImportError as e:
            training_state.error = f"Missing required packages: {e}. Run: pip install peft accelerate"
            training_state.status_message = training_state.error
            return
        
        training_state.status_message = "Loading base model..."
        
        # Get dataset images
        images_data = self.dataset_manager.get_images(config.dataset_name)
        if len(images_data) < 3:
            training_state.error = "Need at least 3 images to train"
            training_state.status_message = training_state.error
            return
        
        training_state.status_message = f"Found {len(images_data)} training images"
        
        # Calculate steps
        steps_per_epoch = len(images_data) // config.train_batch_size
        if config.max_train_steps:
            training_state.total_steps = config.max_train_steps
        else:
            training_state.total_steps = steps_per_epoch * config.num_train_epochs
        training_state.total_epochs = config.num_train_epochs
        
        # Load model components
        training_state.status_message = "Loading model components..."
        
        try:
            # Load pipeline to get components
            pipe = StableDiffusionPipeline.from_pretrained(
                config.base_model,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            unet = pipe.unet
            text_encoder = pipe.text_encoder
            vae = pipe.vae
            tokenizer = pipe.tokenizer
            noise_scheduler = DDPMScheduler.from_pretrained(
                config.base_model,
                subfolder="scheduler",
                cache_dir=self.cache_dir
            )
            
            # Move to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet.to(device)
            text_encoder.to(device)
            vae.to(device)
            
            # Freeze VAE and text encoder
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)
            
            # Apply LoRA to UNet
            training_state.status_message = "Applying LoRA configuration..."
            
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=config.lora_dropout
            )
            
            unet = get_peft_model(unet, lora_config)
            
            # Resume from existing LoRA if specified
            if config.resume_from and os.path.exists(config.resume_from):
                training_state.status_message = f"Loading existing LoRA weights from {config.resume_from}..."
                logger.info(f"Resuming training from: {config.resume_from}")
                try:
                    from safetensors.torch import load_file
                    lora_weights_path = os.path.join(config.resume_from, "adapter_model.safetensors")
                    if os.path.exists(lora_weights_path):
                        state_dict = load_file(lora_weights_path)
                        # Load weights into the PEFT model
                        unet.load_state_dict(state_dict, strict=False)
                        logger.info("Successfully loaded existing LoRA weights for continued training")
                except Exception as e:
                    logger.warning(f"Could not load existing LoRA weights: {e}. Starting fresh.")
            
            unet.print_trainable_parameters()
            
            # Enable gradient checkpointing
            if config.gradient_checkpointing:
                unet.enable_gradient_checkpointing()
            
            # Optimizer
            if config.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(
                        unet.parameters(),
                        lr=config.learning_rate
                    )
                except ImportError:
                    optimizer = torch.optim.AdamW(
                        unet.parameters(),
                        lr=config.learning_rate
                    )
            else:
                optimizer = torch.optim.AdamW(
                    unet.parameters(),
                    lr=config.learning_rate
                )
            
            # Simple dataset
            class TrainingDataset(Dataset):
                def __init__(self, images_data, tokenizer, vae, resolution, trigger_word):
                    self.images_data = images_data
                    self.tokenizer = tokenizer
                    self.vae = vae
                    self.resolution = resolution
                    self.trigger_word = trigger_word
                
                def __len__(self):
                    return len(self.images_data)
                
                def __getitem__(self, idx):
                    item = self.images_data[idx]
                    
                    # Load and preprocess image
                    image = Image.open(item["path"]).convert("RGB")
                    image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                    
                    # Convert to tensor using numpy (no torchvision needed)
                    import numpy as np
                    img_array = np.array(image).astype(np.float32) / 255.0  # [0, 1]
                    img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
                    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                    pixel_values = torch.from_numpy(img_array)
                    
                    # Prepare caption with trigger word
                    caption = item.get("caption", "")
                    if self.trigger_word and self.trigger_word not in caption:
                        caption = f"{self.trigger_word}, {caption}" if caption else self.trigger_word
                    
                    # Tokenize
                    tokens = self.tokenizer(
                        caption,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    return {
                        "pixel_values": pixel_values,
                        "input_ids": tokens.input_ids.squeeze(0)
                    }
            
            dataset = TrainingDataset(images_data, tokenizer, vae, config.resolution, config.trigger_word)
            dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
            
            # Training loop
            training_state.status_message = "Starting training..."
            unet.train()
            global_step = 0
            
            for epoch in range(config.num_train_epochs):
                training_state.current_epoch = epoch + 1
                
                for batch in dataloader:
                    if self._stop_requested:
                        training_state.status_message = "Training stopped by user"
                        break
                    
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                    input_ids = batch["input_ids"].to(device)
                    
                    # Encode images to latent space
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    
                    # Sample timesteps
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, 
                        (latents.shape[0],), device=device
                    ).long()
                    
                    # Add noise to latents
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get text embeddings
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(input_ids)[0]
                    
                    # Predict noise
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample
                    
                    # Calculate loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    
                    # Backprop
                    loss.backward()
                    
                    # Gradient accumulation
                    if (global_step + 1) % config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    global_step += 1
                    training_state.current_step = global_step
                    training_state.loss = loss.item()
                    training_state.status_message = f"Epoch {epoch+1}/{config.num_train_epochs} | Step {global_step}/{training_state.total_steps} | Loss: {loss.item():.4f}"
                    
                    if progress_callback:
                        progress_callback(training_state)
                    
                    # Save checkpoint
                    if global_step % config.save_steps == 0:
                        checkpoint_path = LORA_DIR / f"{config.output_name}_step{global_step}.safetensors"
                        unet.save_pretrained(str(checkpoint_path.parent), safe_serialization=True)
                    
                    if config.max_train_steps and global_step >= config.max_train_steps:
                        break
                
                if self._stop_requested or (config.max_train_steps and global_step >= config.max_train_steps):
                    break
            
            # Save final model
            if not self._stop_requested:
                training_state.status_message = "Saving LoRA..."
                
                # Save to a temp directory first to avoid file lock issues
                # (safetensors memory-maps files, causing issues on Windows when overwriting)
                import tempfile
                import shutil
                
                temp_dir = Path(tempfile.mkdtemp(prefix="lora_train_"))
                temp_output = temp_dir / config.output_name
                
                try:
                    # Save to temp location
                    unet.save_pretrained(str(temp_output), safe_serialization=True)
                    
                    # Clean up the model to release file handles
                    del unet
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # Now move to final location
                    final_output = LORA_DIR / config.output_name
                    if final_output.exists():
                        shutil.rmtree(final_output)
                    shutil.move(str(temp_output), str(final_output))
                    
                    # Save metadata
                    lora_metadata = {
                        "trigger_word": config.trigger_word,
                        "base_model": config.base_model,
                        "training_steps": global_step,
                        "lora_rank": config.lora_rank,
                        "lora_alpha": config.lora_alpha,
                        "description": config.description,
                        "created_at": datetime.now().isoformat()
                    }
                    with open(final_output / "lora_metadata.json", "w") as f:
                        json.dump(lora_metadata, f, indent=2)
                    
                    output_path = final_output / "adapter_model.safetensors"
                    
                finally:
                    # Clean up temp dir
                    if temp_dir.exists():
                        try:
                            shutil.rmtree(temp_dir)
                        except:
                            pass
                
                # Register LoRA
                lora_info = LoRAInfo(
                    name=config.output_name,
                    base_model=config.base_model,
                    trigger_word=config.trigger_word,
                    created_at=datetime.now().isoformat(),
                    training_steps=global_step,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    description=config.description
                )
                self.lora_manager.register_lora(lora_info)
                
                training_state.completed = True
                training_state.output_path = str(output_path)
                training_state.status_message = f"✅ Training complete! LoRA saved to {config.output_name}"
            
            # Cleanup
            try:
                del vae, text_encoder, pipe
            except:
                pass
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_state.error = str(e)
            training_state.status_message = f"Training failed: {e}"
            raise


# Global instances
dataset_manager = DatasetManager()
lora_manager = LoRAManager()
trainer = LoRATrainer()


def get_dataset_manager() -> DatasetManager:
    return dataset_manager


def get_lora_manager() -> LoRAManager:
    return lora_manager


def get_trainer() -> LoRATrainer:
    return trainer
