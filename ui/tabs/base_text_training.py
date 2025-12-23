"""
Base Text Training UI Components for KVGenius Flet App
Shared UI components used by both TextLoRATrainingTab and CardLoRATrainingTab.
Keeps things DRY - both tabs use the same underlying training logic.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from abc import ABC, abstractmethod

import flet as ft
from flet import (
    Page, Container, Column, Row, Text, TextField,
    ElevatedButton, IconButton, Card, Dropdown, Slider,
    ProgressBar, ProgressRing, ListView, Checkbox,
    Icons, Colors, FontWeight, ScrollMode,
    padding, border_radius, alignment,
    CrossAxisAlignment, MainAxisAlignment,
    dropdown, Tabs, Tab
)

from src.training.text_lora_trainer import (
    TextLoRAType, TextTrainingConfig, TextTrainingExample,
    TextLoRATrainer, TextLoRAManager, TextTrainingState,
    get_training_state, get_text_lora_trainer, get_text_lora_manager
)

logger = logging.getLogger(__name__)


class BaseTextTrainingTab(ABC):
    """
    Abstract base class for text LoRA training tabs.
    Provides shared UI components and training logic.
    
    Subclasses implement:
    - get_lora_type() - Return the TextLoRAType for this tab
    - get_tab_title() - Return the tab title
    - get_tab_icon() - Return the tab icon
    - build_data_input_section() - Build the data input UI specific to this type
    - get_training_examples() - Extract training examples from the UI
    """
    
    def __init__(self, page: Page):
        self.page = page
        self.trainer = get_text_lora_trainer(self.get_lora_type())
        self.manager = get_text_lora_manager(self.get_lora_type())
        self._progress_timer = None
        self._build_common_ui()
    
    @abstractmethod
    def get_lora_type(self) -> TextLoRAType:
        """Return the TextLoRAType for this training tab."""
        pass
    
    @abstractmethod
    def get_tab_title(self) -> str:
        """Return the title for this tab."""
        pass
    
    @abstractmethod
    def get_tab_icon(self) -> str:
        """Return the icon emoji for this tab."""
        pass
    
    @abstractmethod
    def build_data_input_section(self) -> Container:
        """Build the data input section specific to this training type."""
        pass
    
    @abstractmethod
    def get_training_examples(self) -> List[TextTrainingExample]:
        """Extract training examples from the UI inputs."""
        pass
    
    def _build_common_ui(self):
        """Build common UI components shared by all training tabs."""
        
        # === Output Configuration ===
        self.output_name_field = TextField(
            label="LoRA Name",
            hint_text="my_custom_lora",
            width=300,
            on_change=self._validate_form,
        )
        
        self.description_field = TextField(
            label="Description (optional)",
            hint_text="Brief description of what this LoRA does",
            width=400,
            multiline=True,
            min_lines=2,
            max_lines=3,
        )
        
        # === Training Parameters ===
        
        # Epochs slider
        self.epochs_value_text = Text("3 epochs", size=11, width=80)
        self.epochs_slider = Slider(
            min=1, max=10, value=3, divisions=9,
            expand=True,
            on_change=self._on_epochs_change,
        )
        
        # Batch size slider
        self.batch_value_text = Text("Batch: 4", size=11, width=70)
        self.batch_slider = Slider(
            min=1, max=8, value=4, divisions=7,
            expand=True,
            on_change=self._on_batch_change,
        )
        
        # Learning rate slider
        self.lr_value_text = Text("LR: 2e-4", size=11, width=80)
        self.lr_slider = Slider(
            min=1, max=5, value=2, divisions=4,  # Maps to 1e-4, 2e-4, 3e-4, 4e-4, 5e-4
            expand=True,
            on_change=self._on_lr_change,
        )
        
        # LoRA Rank slider
        self.rank_value_text = Text("Rank: 16", size=11, width=70)
        self.rank_slider = Slider(
            min=4, max=64, value=16, divisions=15,
            expand=True,
            on_change=self._on_rank_change,
        )
        
        # LoRA Alpha slider
        self.alpha_value_text = Text("Alpha: 32", size=11, width=75)
        self.alpha_slider = Slider(
            min=8, max=128, value=32, divisions=15,
            expand=True,
            on_change=self._on_alpha_change,
        )
        
        # Max sequence length
        self.seq_length_dropdown = Dropdown(
            label="Max Sequence Length",
            width=180,
            options=[
                dropdown.Option("256", "256 tokens"),
                dropdown.Option("512", "512 tokens"),
                dropdown.Option("1024", "1024 tokens"),
                dropdown.Option("2048", "2048 tokens"),
            ],
            value="512",
        )
        
        # Use 8-bit quantization
        self.use_8bit_checkbox = Checkbox(
            label="Use 8-bit quantization (saves VRAM)",
            value=True,
        )
        
        # Gradient checkpointing
        self.grad_checkpoint_checkbox = Checkbox(
            label="Gradient checkpointing (saves VRAM, slower)",
            value=True,
        )
        
        # === Progress & Status ===
        self.progress_bar = ProgressBar(
            width=400,
            value=0,
            color=Colors.GREEN_400,
            bgcolor=Colors.GREY_800,
        )
        
        self.status_text = Text(
            "Ready to train",
            size=12,
            color=Colors.GREY_400,
        )
        
        self.loss_text = Text(
            "",
            size=11,
            color=Colors.BLUE_400,
        )
        
        self.epoch_text = Text(
            "",
            size=11,
            color=Colors.GREY_500,
        )
        
        # === Action Buttons ===
        self.start_btn = ElevatedButton(
            "🚀 Start Training",
            icon=Icons.PLAY_ARROW,
            on_click=self._start_training,
            disabled=True,  # Enable when form is valid
            bgcolor=Colors.GREEN_700,
            color=Colors.WHITE,
        )
        
        self.stop_btn = ElevatedButton(
            "⏹️ Stop",
            icon=Icons.STOP,
            on_click=self._stop_training,
            visible=False,
            bgcolor=Colors.RED_700,
            color=Colors.WHITE,
        )
        
        # === Trained LoRAs List ===
        self.lora_list = ListView(
            spacing=8,
            padding=padding.all(10),
            expand=True,
        )
        
        self.refresh_btn = IconButton(
            icon=Icons.REFRESH,
            tooltip="Refresh LoRA list",
            on_click=lambda e: self._refresh_loras(),
        )
        
        # Training active indicator
        self.training_indicator = Row([
            ProgressRing(width=16, height=16, stroke_width=2, color=Colors.GREEN_400),
            Text("Training in progress...", size=11, color=Colors.GREEN_400),
        ], spacing=8, visible=False)
    
    def _on_epochs_change(self, e):
        val = int(self.epochs_slider.value)
        self.epochs_value_text.value = f"{val} epochs"
        self.page.update()
    
    def _on_batch_change(self, e):
        val = int(self.batch_slider.value)
        self.batch_value_text.value = f"Batch: {val}"
        self.page.update()
    
    def _on_lr_change(self, e):
        val = int(self.lr_slider.value)
        self.lr_value_text.value = f"LR: {val}e-4"
        self.page.update()
    
    def _on_rank_change(self, e):
        val = int(self.rank_slider.value)
        self.rank_value_text.value = f"Rank: {val}"
        self.page.update()
    
    def _on_alpha_change(self, e):
        val = int(self.alpha_slider.value)
        self.alpha_value_text.value = f"Alpha: {val}"
        self.page.update()
    
    def _validate_form(self, e=None):
        """Validate the form and enable/disable start button."""
        name_valid = bool(self.output_name_field.value and self.output_name_field.value.strip())
        examples = self.get_training_examples()
        has_examples = len(examples) >= 3
        
        self.start_btn.disabled = not (name_valid and has_examples)
        try:
            self.page.update()
        except:
            pass
    
    def _build_config_section(self) -> Container:
        """Build the training configuration section."""
        return Card(
            content=Container(
                content=Column([
                    Text("⚙️ Training Configuration", size=14, weight=FontWeight.BOLD),
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    # Output name and description
                    Row([
                        self.output_name_field,
                        self.description_field,
                    ], spacing=15, wrap=True),
                    
                    ft.Divider(height=10, color=Colors.GREY_800),
                    
                    # Training parameters
                    Text("Training Parameters", size=12, weight=FontWeight.W_500),
                    Row([
                        Column([
                            Row([self.epochs_value_text, self.epochs_slider]),
                        ], expand=True),
                        Column([
                            Row([self.batch_value_text, self.batch_slider]),
                        ], expand=True),
                        Column([
                            Row([self.lr_value_text, self.lr_slider]),
                        ], expand=True),
                    ], spacing=15),
                    
                    ft.Divider(height=10, color=Colors.GREY_800),
                    
                    # LoRA parameters
                    Text("LoRA Parameters", size=12, weight=FontWeight.W_500),
                    Row([
                        Column([
                            Row([self.rank_value_text, self.rank_slider]),
                        ], expand=True),
                        Column([
                            Row([self.alpha_value_text, self.alpha_slider]),
                        ], expand=True),
                        self.seq_length_dropdown,
                    ], spacing=15),
                    
                    ft.Divider(height=10, color=Colors.GREY_800),
                    
                    # Optimization options
                    Row([
                        self.use_8bit_checkbox,
                        self.grad_checkpoint_checkbox,
                    ], spacing=20),
                    
                ], spacing=8),
                padding=padding.all(15),
            ),
        )
    
    def _build_progress_section(self) -> Container:
        """Build the progress/status section."""
        return Card(
            content=Container(
                content=Column([
                    Row([
                        Text("📊 Training Progress", size=14, weight=FontWeight.BOLD),
                        Container(expand=True),
                        self.training_indicator,
                    ]),
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    self.progress_bar,
                    Row([
                        self.status_text,
                        Container(expand=True),
                        self.epoch_text,
                        self.loss_text,
                    ], spacing=15),
                    
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    
                    Row([
                        self.start_btn,
                        self.stop_btn,
                    ], spacing=10),
                    
                ], spacing=8),
                padding=padding.all(15),
            ),
        )
    
    def _build_loras_section(self) -> Container:
        """Build the trained LoRAs list section."""
        return Card(
            content=Container(
                content=Column([
                    Row([
                        Text("📦 Trained LoRAs", size=14, weight=FontWeight.BOLD),
                        Container(expand=True),
                        self.refresh_btn,
                    ]),
                    ft.Divider(height=10, color=Colors.TRANSPARENT),
                    Container(
                        content=self.lora_list,
                        height=200,
                    ),
                ], spacing=8),
                padding=padding.all(15),
            ),
        )
    
    def _refresh_loras(self):
        """Refresh the list of trained LoRAs."""
        self.lora_list.controls.clear()
        
        loras = self.manager.list_loras()
        
        if not loras:
            self.lora_list.controls.append(
                Container(
                    content=Text(
                        "No trained LoRAs yet. Train one above!",
                        color=Colors.GREY_500,
                        italic=True,
                    ),
                    padding=padding.all(10),
                )
            )
        else:
            for lora in loras:
                self.lora_list.controls.append(
                    self._build_lora_card(lora)
                )
        
        self.page.update()
    
    def _build_lora_card(self, lora) -> Container:
        """Build a card for a trained LoRA."""
        return Container(
            content=Row([
                Column([
                    Text(lora.name, weight=FontWeight.BOLD, size=13),
                    Text(lora.description or "No description", size=11, color=Colors.GREY_400),
                    Row([
                        Container(
                            content=Text(f"Rank: {lora.rank}", size=10),
                            bgcolor=Colors.BLUE_900,
                            padding=padding.symmetric(horizontal=6, vertical=2),
                            border_radius=border_radius.all(4),
                        ),
                        Container(
                            content=Text(f"{lora.training_steps} steps", size=10),
                            bgcolor=Colors.GREEN_900,
                            padding=padding.symmetric(horizontal=6, vertical=2),
                            border_radius=border_radius.all(4),
                        ),
                        Container(
                            content=Text(f"{lora.file_size_mb:.1f} MB", size=10),
                            bgcolor=Colors.PURPLE_900,
                            padding=padding.symmetric(horizontal=6, vertical=2),
                            border_radius=border_radius.all(4),
                        ),
                    ], spacing=5),
                ], expand=True, spacing=3),
                IconButton(
                    icon=Icons.DELETE_OUTLINE,
                    icon_color=Colors.RED_400,
                    tooltip="Delete LoRA",
                    on_click=lambda e, n=lora.name: self._delete_lora(n),
                ),
            ], spacing=10),
            padding=padding.all(10),
            bgcolor=Colors.with_opacity(0.05, Colors.WHITE),
            border_radius=border_radius.all(8),
        )
    
    def _delete_lora(self, name: str):
        """Delete a trained LoRA."""
        def do_delete(e):
            self.manager.delete_lora(name)
            dialog.open = False
            self._refresh_loras()
            self.page.snack_bar = ft.SnackBar(content=Text(f"🗑️ Deleted: {name}"))
            self.page.snack_bar.open = True
            self.page.update()
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = ft.AlertDialog(
            title=Text("Delete LoRA?"),
            content=Text(f"Are you sure you want to delete '{name}'?\nThis cannot be undone."),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.TextButton("Delete", on_click=do_delete),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()
    
    def _get_training_config(self) -> TextTrainingConfig:
        """Build a TrainingConfig from the current UI state."""
        examples = self.get_training_examples()
        
        return TextTrainingConfig(
            output_name=self.output_name_field.value.strip(),
            lora_type=self.get_lora_type().value,
            description=self.description_field.value.strip() if self.description_field.value else "",
            training_examples=examples,
            num_train_epochs=int(self.epochs_slider.value),
            train_batch_size=int(self.batch_slider.value),
            learning_rate=int(self.lr_slider.value) * 1e-4,
            lora_rank=int(self.rank_slider.value),
            lora_alpha=int(self.alpha_slider.value),
            use_8bit=self.use_8bit_checkbox.value,
            gradient_checkpointing=self.grad_checkpoint_checkbox.value,
            max_seq_length=int(self.seq_length_dropdown.value),
        )
    
    def _start_training(self, e):
        """Start the training process."""
        config = self._get_training_config()
        
        # Validate
        valid, error = self.trainer.validate_config(config)
        if not valid:
            self.page.snack_bar = ft.SnackBar(
                content=Text(f"❌ {error}"),
                bgcolor=Colors.RED_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        
        # Update UI state
        self.start_btn.visible = False
        self.stop_btn.visible = True
        self.training_indicator.visible = True
        self.progress_bar.value = 0
        self.status_text.value = "Starting..."
        self.page.update()
        
        # Start training
        success = self.trainer.start_training(
            config,
            progress_callback=self._on_training_progress
        )
        
        if not success:
            self.start_btn.visible = True
            self.stop_btn.visible = False
            self.training_indicator.visible = False
            state = self.trainer.get_state()
            self.status_text.value = state.error or "Failed to start training"
            self.page.update()
            return
        
        # Start progress polling
        self._start_progress_polling()
    
    def _stop_training(self, e):
        """Stop the training process."""
        self.trainer.request_stop()
        self.status_text.value = "Stopping..."
        self.page.update()
    
    def _on_training_progress(self, state: TextTrainingState):
        """Callback for training progress updates."""
        # This runs in training thread, we need to update UI carefully
        pass
    
    def _start_progress_polling(self):
        """Start polling for training progress."""
        def poll():
            state = self.trainer.get_state()
            
            self.progress_bar.value = state.progress_pct / 100
            self.status_text.value = state.status_message
            self.epoch_text.value = f"Epoch {state.current_epoch}/{state.total_epochs}" if state.total_epochs else ""
            self.loss_text.value = f"Loss: {state.loss:.4f}" if state.loss > 0 else ""
            
            try:
                self.page.update()
            except:
                pass
            
            if state.is_training:
                # Continue polling
                self._progress_timer = threading.Timer(0.5, poll)
                self._progress_timer.start()
            else:
                # Training finished
                self.start_btn.visible = True
                self.stop_btn.visible = False
                self.training_indicator.visible = False
                
                if state.completed:
                    self.page.snack_bar = ft.SnackBar(
                        content=Text(f"✅ Training complete! LoRA saved."),
                        bgcolor=Colors.GREEN_700,
                    )
                    self.page.snack_bar.open = True
                    self._refresh_loras()
                elif state.error:
                    self.page.snack_bar = ft.SnackBar(
                        content=Text(f"❌ Training failed: {state.error}"),
                        bgcolor=Colors.RED_700,
                    )
                    self.page.snack_bar.open = True
                
                try:
                    self.page.update()
                except:
                    pass
        
        poll()
    
    def build(self) -> Container:
        """Build the complete training tab."""
        # Header
        header = Container(
            content=Column([
                Row([
                    Text(f"{self.get_tab_icon()} {self.get_tab_title()}", size=20, weight=FontWeight.BOLD),
                ]),
                Text(
                    self._get_header_description(),
                    size=12, color=Colors.GREY_400
                ),
            ]),
            padding=padding.all(20),
        )
        
        # Build layout with two columns
        left_column = Column([
            self.build_data_input_section(),
            self._build_config_section(),
        ], spacing=15, expand=True)
        
        right_column = Column([
            self._build_progress_section(),
            self._build_loras_section(),
        ], spacing=15, width=450)
        
        main_content = Row([
            left_column,
            right_column,
        ], spacing=20, expand=True, vertical_alignment=CrossAxisAlignment.START)
        
        # Refresh LoRAs on build
        self._refresh_loras()
        
        return Container(
            content=Column([
                header,
                Container(
                    content=main_content,
                    padding=padding.symmetric(horizontal=20),
                    expand=True,
                ),
            ], scroll=ScrollMode.AUTO, expand=True),
            expand=True,
        )
    
    def _get_header_description(self) -> str:
        """Get the description text for the header."""
        lora_type = self.get_lora_type()
        if lora_type == TextLoRAType.CARD_GENERATOR:
            return "Train custom card themes and humor styles for card generation"
        else:
            return "Train custom personas and chat styles for text models"
