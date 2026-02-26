# UI Tabs Package
from .image_gen import ImageGenTab
from .gallery import GalleryTab
from .prompts import PromptLibraryTab
from .model_manager import ModelManagerTab
from .chat import ChatTab
from .settings import SettingsTab
from .lora_manager import LoRAManagerTab
from .dataset_manager import DatasetManagerTab
from .lora_training import LoRATrainingTab
from .text_lora_training import TextLoRATrainingTab
from .card_lora_training import CardLoRATrainingTab
from .model_hub import ModelHubTab
from .card_generator import CardGeneratorTab

__all__ = [
    'ImageGenTab',
    'GalleryTab', 
    'PromptLibraryTab',
    'ModelManagerTab',
    'ChatTab',
    'SettingsTab',
    'LoRAManagerTab',
    'DatasetManagerTab',
    'LoRATrainingTab',
    'TextLoRATrainingTab',
    'CardLoRATrainingTab',
    'ModelHubTab',
    'CardGeneratorTab',
]
