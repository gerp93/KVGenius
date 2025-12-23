"""
Model Hub Tab for KVGenius Flet App
Combines Model Manager, LoRA Manager, Dataset Manager, and LoRA Training
into a single tab with sub-tabs for better organization.
"""

import logging
from flet import (
    Page, Container, Column, Row, Text, Tab, Tabs,
    Icons, Colors, FontWeight, padding, border_radius,
)

logger = logging.getLogger(__name__)

# Import the individual tab classes
from .model_manager import ModelManagerTab
from .lora_manager import LoRAManagerTab
from .dataset_manager import DatasetManagerTab
from .lora_training import LoRATrainingTab
from .text_lora_training import TextLoRATrainingTab
from .card_lora_training import CardLoRATrainingTab


class ModelHubTab:
    """
    Model Hub - Combined tab for all model-related functionality.
    Contains sub-tabs for:
    - Models (download/manage image & chat models)
    - LoRAs (browse/import/edit LoRA files)
    - Datasets (manage training datasets)
    - Training (configure and run image LoRA training)
    - Text Training (train text/chat model LoRAs)
    - Card Training (train card generation LoRAs)
    """
    
    def __init__(self, page: Page):
        self.page = page
        
        # Create sub-tab instances
        self.model_manager = ModelManagerTab(page)
        self.lora_manager = LoRAManagerTab(page)
        self.dataset_manager = DatasetManagerTab(page)
        self.lora_training = LoRATrainingTab(page)
        self.text_lora_training = TextLoRATrainingTab(page)
        self.card_lora_training = CardLoRATrainingTab(page)
    
    def build(self) -> Container:
        """Build the Model Hub tab with sub-tabs."""
        
        # Create sub-tabs
        sub_tabs = Tabs(
            selected_index=0,
            animation_duration=200,
            tabs=[
                Tab(
                    text="📦 Models",
                    icon=Icons.DOWNLOAD,
                    content=self.model_manager.build(),
                ),
                Tab(
                    text="🎨 LoRAs",
                    icon=Icons.AUTO_AWESOME,
                    content=self.lora_manager.build(),
                ),
                Tab(
                    text="📂 Datasets",
                    icon=Icons.FOLDER_COPY,
                    content=self.dataset_manager.build(),
                ),
                Tab(
                    text="🎓 Train Image",
                    icon=Icons.MODEL_TRAINING,
                    content=self.lora_training.build(),
                ),
                Tab(
                    text="💬 Train Text",
                    icon=Icons.CHAT,
                    content=self.text_lora_training.build(),
                ),
                Tab(
                    text="🃏 Train Cards",
                    icon=Icons.STYLE,
                    content=self.card_lora_training.build(),
                ),
            ],
            expand=True,
        )
        
        return Container(
            content=Column([
                # Header
                Container(
                    content=Row([
                        Text("🧠 Model Hub", size=20, weight=FontWeight.BOLD),
                        Text("Manage models, LoRAs, datasets, and training", 
                             size=12, color=Colors.GREY_400),
                    ], spacing=15),
                    padding=padding.only(left=15, top=10, bottom=5),
                ),
                # Sub-tabs
                sub_tabs,
            ], spacing=0),
            expand=True,
        )
