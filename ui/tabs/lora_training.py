"""
LoRA Training Tab for KVGenius Flet App
Provides UI for training LoRAs with configurable parameters.
"""

import os
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Callable
from datetime import datetime

import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextField, 
    ElevatedButton, IconButton, TextButton, Card, Image,
    Icons, Colors, FontWeight, ScrollMode, AlertDialog, 
    SnackBar, ListView, Dropdown, dropdown, Slider,
    ProgressBar, ProgressRing, Divider, Checkbox,
    padding, border_radius, alignment,
    ControlState, ButtonStyle, MainAxisAlignment, CrossAxisAlignment,
)

logger = logging.getLogger(__name__)

# Import training components
from src.training.lora_trainer import (
    TrainingConfig,
    training_state,
    get_trainer,
    get_dataset_manager,
    get_lora_manager,
    LORA_DIR,
    DATASETS_DIR,
)


class LoRATrainingTab:
    """LoRA Training - configure and run LoRA training jobs."""
    
    # Available base models
    BASE_MODELS = {
        "sd15": {
            "name": "Stable Diffusion 1.5",
            "id": "runwayml/stable-diffusion-v1-5",
            "resolution": 512,
        },
        "sd21": {
            "name": "Stable Diffusion 2.1",
            "id": "stabilityai/stable-diffusion-2-1",
            "resolution": 768,
        },
        "sdxl": {
            "name": "SDXL 1.0",
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "resolution": 1024,
        },
    }
    
    def __init__(self, page: Page):
        self.page = page
        self.trainer = get_trainer()
        self.dataset_manager = get_dataset_manager()
        self.lora_manager = get_lora_manager()
        self._progress_thread = None
        self._build_ui()
    
    def _build_ui(self):
        """Build all UI components."""
        # === Dataset Selection ===
        self.dataset_dropdown = Dropdown(
            label="Training Dataset",
            width=300,
            options=self._get_dataset_options(),
            on_change=self._on_dataset_change,
        )
        
        self.dataset_info = Text("", size=11, color=Colors.GREY_400)
        
        self.refresh_datasets_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh datasets",
            on_click=lambda e: self._refresh_datasets(),
        )
        
        # === Output Configuration ===
        self.output_name = TextField(
            label="LoRA Output Name",
            hint_text="my_custom_lora",
            width=300,
        )
        
        self.trigger_word = TextField(
            label="Trigger Word(s)",
            hint_text="e.g., sks person, twilek",
            width=300,
        )
        
        self.description = TextField(
            label="Description",
            hint_text="Optional description for the LoRA",
            width=400,
            multiline=True,
            min_lines=2,
            max_lines=4,
        )
        
        # === Base Model Selection ===
        self.base_model_dropdown = Dropdown(
            label="Base Model",
            width=300,
            options=[
                dropdown.Option("sd15", "🎨 Stable Diffusion 1.5 (512px)"),
                dropdown.Option("sdxl", "🎨 SDXL 1.0 (1024px)"),
            ],
            value="sd15",
            on_change=self._on_base_model_change,
        )
        
        # === Resume from existing LoRA ===
        self.resume_checkbox = Checkbox(
            label="Resume/Continue from existing LoRA",
            value=False,
            on_change=self._on_resume_change,
        )
        
        self.resume_lora_dropdown = Dropdown(
            label="Resume from",
            width=300,
            options=self._get_lora_options(),
            visible=False,
        )
        
        # === Training Parameters ===
        # Epochs
        self.epochs_value = Text("100", size=11, width=60)
        self.epochs_slider = Slider(
            min=10, max=500, value=100,
            divisions=49,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.epochs_value, "{}", int(self.epochs_slider.value)),
        )
        
        # Learning Rate
        self.lr_value = Text("1e-4", size=11, width=60)
        self.lr_slider = Slider(
            min=1, max=10, value=4,  # Maps to 1e-5 to 1e-3
            divisions=9,
            expand=True,
            on_change=self._on_lr_change,
        )
        
        # Batch Size
        self.batch_value = Text("1", size=11, width=60)
        self.batch_slider = Slider(
            min=1, max=4, value=1,
            divisions=3,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.batch_value, "{}", int(self.batch_slider.value)),
        )
        
        # Gradient Accumulation
        self.grad_accum_value = Text("4", size=11, width=60)
        self.grad_accum_slider = Slider(
            min=1, max=16, value=4,
            divisions=15,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.grad_accum_value, "{}", int(self.grad_accum_slider.value)),
        )
        
        # LoRA Rank
        self.rank_value = Text("16", size=11, width=60)
        self.rank_slider = Slider(
            min=4, max=128, value=16,
            divisions=31,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.rank_value, "{}", int(self.rank_slider.value)),
        )
        
        # LoRA Alpha
        self.alpha_value = Text("32", size=11, width=60)
        self.alpha_slider = Slider(
            min=8, max=256, value=32,
            divisions=31,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.alpha_value, "{}", int(self.alpha_slider.value)),
        )
        
        # Resolution
        self.resolution_value = Text("512", size=11, width=60)
        self.resolution_slider = Slider(
            min=256, max=1024, value=512,
            divisions=3,
            expand=True,
            on_change=lambda e: self._update_slider_label(self.resolution_value, "{}", int(self.resolution_slider.value)),
        )
        
        # Checkboxes
        self.use_8bit_adam = Checkbox(label="Use 8-bit Adam (saves VRAM)", value=True)
        self.gradient_checkpointing = Checkbox(label="Gradient Checkpointing", value=True)
        self.center_crop = Checkbox(label="Center Crop Images", value=True)
        
        # === Training Controls ===
        self.start_btn = ElevatedButton(
            "▶️ Start Training",
            icon=Icons.PLAY_ARROW,
            bgcolor=Colors.GREEN_700,
            color=Colors.WHITE,
            on_click=self._start_training,
        )
        
        self.stop_btn = ElevatedButton(
            "⏹️ Stop Training",
            icon=Icons.STOP,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
            on_click=self._stop_training,
            disabled=True,
        )
        
        # === Progress Section ===
        self.progress_bar = ProgressBar(value=0, visible=False)
        self.status_text = Text("Ready", size=12)
        self.progress_details = Text("", size=11, color=Colors.GREY_400)
        self.loss_text = Text("", size=11, color=Colors.BLUE_400)
        
        # Training log
        self.training_log = ListView(
            spacing=5,
            padding=padding.all(10),
            auto_scroll=True,
            expand=True,
        )
        
        # Initialize slider labels
        self._on_lr_change(None)
    
    def _update_slider_label(self, label_control: Text, format_str: str, value):
        """Update slider value label."""
        label_control.value = format_str.format(value)
        try:
            self.page.update()
        except:
            pass
    
    def _on_lr_change(self, e):
        """Update learning rate label (logarithmic scale)."""
        exp = int(self.lr_slider.value)
        lr = 10 ** (-exp)
        self.lr_value.value = f"{lr:.0e}"
        try:
            self.page.update()
        except:
            pass
    
    def _get_dataset_options(self) -> List[dropdown.Option]:
        """Get dropdown options for datasets."""
        options = []
        datasets = self.dataset_manager.list_datasets()
        for name in datasets:
            images = self.dataset_manager.get_images(name)
            options.append(dropdown.Option(name, f"📁 {name} ({len(images)} images)"))
        return options
    
    def _get_lora_options(self) -> List[dropdown.Option]:
        """Get dropdown options for existing LoRAs."""
        options = []
        loras = self.lora_manager.list_loras()
        for lora in loras:
            options.append(dropdown.Option(lora.name, f"🎨 {lora.name}"))
        return options
    
    def _refresh_datasets(self):
        """Refresh the dataset dropdown."""
        self.dataset_dropdown.options = self._get_dataset_options()
        self.resume_lora_dropdown.options = self._get_lora_options()
        self.page.update()
    
    def _on_dataset_change(self, e):
        """Handle dataset selection change."""
        if not self.dataset_dropdown.value:
            self.dataset_info.value = ""
            return
        
        images = self.dataset_manager.get_images(self.dataset_dropdown.value)
        self.dataset_info.value = f"📷 {len(images)} images available for training"
        
        # Auto-suggest output name if empty
        if not self.output_name.value:
            self.output_name.value = f"{self.dataset_dropdown.value}_lora"
        
        self.page.update()
    
    def _on_base_model_change(self, e):
        """Handle base model selection change."""
        model_key = self.base_model_dropdown.value
        if model_key in self.BASE_MODELS:
            model_info = self.BASE_MODELS[model_key]
            self.resolution_slider.value = model_info["resolution"]
            self.resolution_value.value = str(model_info["resolution"])
            self.page.update()
    
    def _on_resume_change(self, e):
        """Toggle resume from existing LoRA."""
        self.resume_lora_dropdown.visible = self.resume_checkbox.value
        self.page.update()
    
    def _validate_config(self) -> Optional[str]:
        """Validate training configuration. Returns error message or None."""
        if not self.dataset_dropdown.value:
            return "Please select a training dataset"
        
        images = self.dataset_manager.get_images(self.dataset_dropdown.value)
        if len(images) < 3:
            return "Need at least 3 images to train"
        
        if not self.output_name.value or not self.output_name.value.strip():
            return "Please enter an output name for the LoRA"
        
        # Check if output already exists
        output_path = LORA_DIR / self.output_name.value.strip()
        if output_path.exists() and not self.resume_checkbox.value:
            return f"LoRA '{self.output_name.value}' already exists. Enable 'Resume' to continue training."
        
        if not self.trigger_word.value or not self.trigger_word.value.strip():
            return "Please enter at least one trigger word"
        
        if self.resume_checkbox.value and not self.resume_lora_dropdown.value:
            return "Please select a LoRA to resume from"
        
        return None
    
    def _log_message(self, message: str, color: str = Colors.GREY_300):
        """Add a message to the training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.controls.append(
            Text(f"[{timestamp}] {message}", size=11, color=color)
        )
        # Keep only last 100 messages
        if len(self.training_log.controls) > 100:
            self.training_log.controls.pop(0)
        try:
            self.page.update()
        except:
            pass
    
    def _start_training(self, e):
        """Start the training process."""
        # Validate
        error = self._validate_config()
        if error:
            self.page.snack_bar = SnackBar(
                content=Text(f"❌ {error}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        # Build config
        model_key = self.base_model_dropdown.value
        base_model_id = self.BASE_MODELS.get(model_key, {}).get("id", "runwayml/stable-diffusion-v1-5")
        
        lr_exp = int(self.lr_slider.value)
        learning_rate = 10 ** (-lr_exp)
        
        resume_path = None
        if self.resume_checkbox.value and self.resume_lora_dropdown.value:
            resume_path = str(LORA_DIR / self.resume_lora_dropdown.value)
        
        config = TrainingConfig(
            dataset_name=self.dataset_dropdown.value,
            output_name=self.output_name.value.strip(),
            base_model=base_model_id,
            trigger_word=self.trigger_word.value.strip(),
            description=self.description.value.strip() if self.description.value else "",
            resume_from=resume_path,
            num_train_epochs=int(self.epochs_slider.value),
            train_batch_size=int(self.batch_slider.value),
            gradient_accumulation_steps=int(self.grad_accum_slider.value),
            learning_rate=learning_rate,
            lora_rank=int(self.rank_slider.value),
            lora_alpha=int(self.alpha_slider.value),
            resolution=int(self.resolution_slider.value),
            center_crop=self.center_crop.value,
            use_8bit_adam=self.use_8bit_adam.value,
            gradient_checkpointing=self.gradient_checkpointing.value,
        )
        
        # Update UI
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.training_log.controls.clear()
        
        self._log_message(f"Starting training: {config.output_name}", Colors.GREEN_400)
        self._log_message(f"Dataset: {config.dataset_name}", Colors.BLUE_400)
        self._log_message(f"Base Model: {config.base_model}", Colors.BLUE_400)
        self._log_message(f"Trigger: {config.trigger_word}", Colors.BLUE_400)
        if resume_path:
            self._log_message(f"Resuming from: {resume_path}", Colors.AMBER_400)
        
        self.page.update()
        
        # Start training
        def progress_callback(state):
            try:
                # Update progress
                if state.total_steps > 0:
                    progress = state.current_step / state.total_steps
                    self.progress_bar.value = progress
                
                self.status_text.value = state.status_message
                self.progress_details.value = f"Epoch {state.current_epoch}/{state.total_epochs} | Step {state.current_step}/{state.total_steps}"
                self.loss_text.value = f"Loss: {state.loss:.4f}" if state.loss > 0 else ""
                
                self.page.update()
            except:
                pass
        
        success = self.trainer.start_training(config, progress_callback)
        
        if not success:
            self._log_message("Training already in progress!", Colors.RED_400)
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            self.page.update()
            return
        
        # Start progress monitoring thread
        def monitor_progress():
            import time
            while training_state.is_training:
                time.sleep(0.5)
                try:
                    if training_state.total_steps > 0:
                        progress = training_state.current_step / training_state.total_steps
                        self.progress_bar.value = progress
                    
                    self.status_text.value = training_state.status_message
                    self.page.update()
                except:
                    pass
            
            # Training finished
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            
            if training_state.completed:
                self._log_message(f"✅ Training completed!", Colors.GREEN_400)
                self._log_message(f"Output: {training_state.output_path}", Colors.GREEN_400)
                self.progress_bar.value = 1.0
                self.page.snack_bar = SnackBar(
                    content=Text(f"✅ LoRA saved: {config.output_name}"),
                )
                self.page.snack_bar.open = True
            elif training_state.error:
                self._log_message(f"❌ Error: {training_state.error}", Colors.RED_400)
                self.page.snack_bar = SnackBar(
                    content=Text(f"❌ Training failed: {training_state.error}"),
                    bgcolor=Colors.RED_700,
                )
                self.page.snack_bar.open = True
            else:
                self._log_message("Training stopped", Colors.AMBER_400)
            
            try:
                self.page.update()
            except:
                pass
        
        self._progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        self._progress_thread.start()
    
    def _stop_training(self, e):
        """Stop the training process."""
        self.trainer.request_stop()
        self._log_message("Stopping training...", Colors.AMBER_400)
        self.stop_btn.disabled = True
        self.page.update()
    
    def build(self) -> Container:
        """Build the LoRA Training tab."""
        self._refresh_datasets()
        
        # Header
        header = Row([
            Text("🎓 LoRA Training", size=18, weight=FontWeight.BOLD),
            Container(expand=True),
            self.refresh_datasets_btn,
        ], alignment=MainAxisAlignment.SPACE_BETWEEN)
        
        # Left panel - Configuration
        config_panel = Container(
            content=Column([
                Text("Training Configuration", size=14, weight=FontWeight.BOLD),
                Divider(),
                
                # Dataset
                Text("Dataset", size=12, weight=FontWeight.BOLD, color=Colors.BLUE_400),
                Row([self.dataset_dropdown, self.refresh_datasets_btn]),
                self.dataset_info,
                
                Container(height=10),
                
                # Output
                Text("Output", size=12, weight=FontWeight.BOLD, color=Colors.BLUE_400),
                self.output_name,
                self.trigger_word,
                self.description,
                
                Container(height=10),
                
                # Base Model
                Text("Base Model", size=12, weight=FontWeight.BOLD, color=Colors.BLUE_400),
                self.base_model_dropdown,
                
                Container(height=5),
                self.resume_checkbox,
                self.resume_lora_dropdown,
                
            ], spacing=8, scroll=ScrollMode.AUTO),
            expand=True,
            padding=padding.all(15),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(10),
        )
        
        # Center panel - Parameters
        params_panel = Container(
            content=Column([
                Text("Training Parameters", size=14, weight=FontWeight.BOLD),
                Divider(),
                
                # Epochs
                Row([Text("Epochs:", size=11, width=120), self.epochs_slider, self.epochs_value]),
                
                # Learning Rate
                Row([Text("Learning Rate:", size=11, width=120), self.lr_slider, self.lr_value]),
                
                # Batch Size
                Row([Text("Batch Size:", size=11, width=120), self.batch_slider, self.batch_value]),
                
                # Gradient Accumulation
                Row([Text("Grad Accumulation:", size=11, width=120), self.grad_accum_slider, self.grad_accum_value]),
                
                Divider(),
                Text("LoRA Parameters", size=12, weight=FontWeight.BOLD, color=Colors.PURPLE_400),
                
                # Rank
                Row([Text("LoRA Rank:", size=11, width=120), self.rank_slider, self.rank_value]),
                
                # Alpha
                Row([Text("LoRA Alpha:", size=11, width=120), self.alpha_slider, self.alpha_value]),
                
                Divider(),
                Text("Image Processing", size=12, weight=FontWeight.BOLD, color=Colors.GREEN_400),
                
                # Resolution
                Row([Text("Resolution:", size=11, width=120), self.resolution_slider, self.resolution_value]),
                
                # Checkboxes
                self.center_crop,
                
                Divider(),
                Text("Optimization", size=12, weight=FontWeight.BOLD, color=Colors.AMBER_400),
                self.use_8bit_adam,
                self.gradient_checkpointing,
                
            ], spacing=8, scroll=ScrollMode.AUTO),
            expand=True,
            padding=padding.all(15),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(10),
        )
        
        # Right panel - Progress & Log
        progress_panel = Container(
            content=Column([
                Text("Training Progress", size=14, weight=FontWeight.BOLD),
                Divider(),
                
                # Controls
                Row([self.start_btn, self.stop_btn], spacing=10),
                
                Container(height=10),
                
                # Status
                self.status_text,
                self.progress_bar,
                self.progress_details,
                self.loss_text,
                
                Container(height=10),
                Divider(),
                
                Text("Training Log", size=12, weight=FontWeight.BOLD),
                Container(
                    content=self.training_log,
                    expand=True,
                    bgcolor=Colors.with_opacity(0.1, Colors.BLACK),
                    border_radius=border_radius.all(8),
                ),
                
            ], spacing=8, expand=True),
            expand=True,
            padding=padding.all(15),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(10),
        )
        
        # Main layout
        return Container(
            content=Column([
                header,
                Row([
                    # Left: Configuration
                    Container(content=config_panel, width=350),
                    # Center: Parameters
                    Container(content=params_panel, width=350),
                    # Right: Progress
                    Container(content=progress_panel, expand=True),
                ], expand=True, spacing=15),
            ], spacing=10),
            padding=padding.all(15),
            expand=True,
        )
