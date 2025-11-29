"""
LoRA Manager - handles discovery, metadata, and loading of LoRA models.
Supports image LoRAs (SD/SDXL), text model LoRAs, and CAH LoRAs.
"""
import os
import json
import logging
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LoRAUseCase(Enum):
    """Use case categories for LoRAs."""
    IMAGE = "image"
    TEXT = "text"
    CAH = "cah"


class LoRABaseModel(Enum):
    """Base model architectures for LoRA compatibility."""
    # Image models
    SD15 = "sd15"
    SDXL = "sdxl"
    # Text models
    MISTRAL = "mistral"
    LLAMA = "llama"
    LLAMA3 = "llama3"
    # Unknown/auto-detect
    UNKNOWN = "unknown"


@dataclass
class LoRAInfo:
    """Information about a LoRA model."""
    name: str
    path: str
    use_case: LoRAUseCase
    base_model: LoRABaseModel
    trigger_words: List[str] = field(default_factory=list)
    compatible_models: List[str] = field(default_factory=list)
    default_weight: float = 0.8
    category: str = "general"
    description: str = ""
    preview_image: Optional[str] = None
    rank: int = 0
    source: str = ""
    
    # Runtime state
    is_loaded: bool = False
    current_weight: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "use_case": self.use_case.value,
            "base_model": self.base_model.value,
            "trigger_words": self.trigger_words,
            "compatible_models": self.compatible_models,
            "default_weight": self.default_weight,
            "category": self.category,
            "description": self.description,
            "preview_image": self.preview_image,
            "rank": self.rank,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "") -> "LoRAInfo":
        """Create from dictionary."""
        return cls(
            name=data.get("name", os.path.basename(path)),
            path=path or data.get("path", ""),
            use_case=LoRAUseCase(data.get("use_case", "image")),
            base_model=LoRABaseModel(data.get("base_model", "unknown")),
            trigger_words=data.get("trigger_words", [data.get("trigger_word", "")]) if data.get("trigger_words") or data.get("trigger_word") else [],
            compatible_models=data.get("compatible_models", []),
            default_weight=data.get("default_weight", 0.8),
            category=data.get("category", "general"),
            description=data.get("description", ""),
            preview_image=data.get("preview_image"),
            rank=data.get("rank", 0),
            source=data.get("source", ""),
        )


class LoRAManager:
    """Manages LoRA discovery, metadata, and loading."""
    
    def __init__(self, base_dir: str = "data/lora_models", config_path: str = "config/lora_presets.yaml"):
        """Initialize LoRA manager.
        
        Args:
            base_dir: Base directory for LoRA files
            config_path: Path to lora_presets.yaml
        """
        self.base_dir = base_dir
        self.config_path = config_path
        
        # Discovered LoRAs by use case
        self.image_loras: Dict[str, LoRAInfo] = {}
        self.text_loras: Dict[str, LoRAInfo] = {}
        self.cah_loras: Dict[str, LoRAInfo] = {}
        
        # Currently loaded LoRAs (for multi-LoRA stacking)
        self.loaded_image_loras: List[str] = []
        self.loaded_text_loras: List[str] = []
        
        # Load config and scan
        self.config = self._load_config()
        self.scan_all()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load lora_presets.yaml configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load LoRA config: {e}")
        return {}
    
    def _detect_base_model_from_adapter_config(self, adapter_config: Dict) -> LoRABaseModel:
        """Detect base model type from adapter_config.json."""
        base_model_path = adapter_config.get("base_model_name_or_path", "")
        
        if base_model_path:
            base_lower = base_model_path.lower()
            if "sdxl" in base_lower or "xl" in base_lower:
                return LoRABaseModel.SDXL
            elif "stable-diffusion" in base_lower or "sd-" in base_lower:
                return LoRABaseModel.SD15
            elif "llama-3" in base_lower or "llama3" in base_lower:
                return LoRABaseModel.LLAMA3
            elif "llama" in base_lower:
                return LoRABaseModel.LLAMA
            elif "mistral" in base_lower:
                return LoRABaseModel.MISTRAL
        
        # Check auto_mapping for diffusers models
        auto_mapping = adapter_config.get("auto_mapping", {})
        parent_library = auto_mapping.get("parent_library", "")
        if "diffusers" in parent_library:
            # Default to SD15 for diffusers LoRAs without explicit base
            return LoRABaseModel.SD15
        
        return LoRABaseModel.UNKNOWN
    
    def _detect_use_case_from_adapter_config(self, adapter_config: Dict) -> LoRAUseCase:
        """Detect use case from adapter_config.json."""
        auto_mapping = adapter_config.get("auto_mapping", {})
        parent_library = auto_mapping.get("parent_library", "")
        base_model_class = auto_mapping.get("base_model_class", "")
        
        # Check for diffusers (image) vs transformers (text)
        if "diffusers" in parent_library or "UNet" in base_model_class:
            return LoRAUseCase.IMAGE
        elif "transformers" in parent_library:
            return LoRAUseCase.TEXT
        
        # Check base_model_name_or_path
        base_model_path = adapter_config.get("base_model_name_or_path", "").lower()
        if any(x in base_model_path for x in ["stable-diffusion", "sdxl", "sd-"]):
            return LoRAUseCase.IMAGE
        elif any(x in base_model_path for x in ["llama", "mistral", "dolphin", "hermes"]):
            return LoRAUseCase.TEXT
        
        # Default to image for unknown
        return LoRAUseCase.IMAGE
    
    def scan_directory(self, directory: str) -> List[LoRAInfo]:
        """Scan a directory for LoRA models.
        
        Looks for:
        - Folders with adapter_config.json (PEFT format)
        - Folders with lora_metadata.json
        - Standalone .safetensors files
        
        Args:
            directory: Directory to scan
        
        Returns:
            List of discovered LoRAInfo objects
        """
        loras = []
        
        if not os.path.exists(directory):
            return loras
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            # Skip non-directories for now (TODO: handle standalone .safetensors)
            if not os.path.isdir(item_path):
                continue
            
            # Check for PEFT format (adapter_config.json)
            adapter_config_path = os.path.join(item_path, "adapter_config.json")
            metadata_path = os.path.join(item_path, "lora_metadata.json")
            
            if os.path.exists(adapter_config_path):
                try:
                    # Load adapter config
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        adapter_config = json.load(f)
                    
                    # Load metadata if exists
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    
                    # Detect base model and use case
                    base_model = self._detect_base_model_from_adapter_config(adapter_config)
                    use_case = self._detect_use_case_from_adapter_config(adapter_config)
                    
                    # Build trigger words list
                    trigger_words = []
                    if metadata.get("trigger_word"):
                        trigger_words = [metadata["trigger_word"]]
                    elif metadata.get("trigger_words"):
                        trigger_words = metadata["trigger_words"]
                    
                    # Look for preview image
                    preview_image = None
                    for preview_name in ["preview.png", "preview.jpg", "preview.jpeg", "thumbnail.png"]:
                        preview_path = os.path.join(item_path, preview_name)
                        if os.path.exists(preview_path):
                            preview_image = preview_path
                            break
                    
                    lora_info = LoRAInfo(
                        name=metadata.get("name", item),
                        path=item_path,
                        use_case=use_case,
                        base_model=base_model,
                        trigger_words=trigger_words,
                        compatible_models=[],
                        default_weight=0.8,
                        category=metadata.get("category", "general"),
                        description=metadata.get("description", ""),
                        preview_image=preview_image,
                        rank=adapter_config.get("r", metadata.get("rank", 0)),
                        source=metadata.get("source", ""),
                    )
                    
                    loras.append(lora_info)
                    logger.debug(f"Found LoRA: {lora_info.name} ({lora_info.use_case.value}, {lora_info.base_model.value})")
                    
                except Exception as e:
                    logger.warning(f"Error scanning LoRA at {item_path}: {e}")
        
        return loras
    
    def scan_all(self):
        """Scan all configured directories for LoRAs."""
        self.image_loras.clear()
        self.text_loras.clear()
        self.cah_loras.clear()
        
        # Get scan paths from config
        settings = self.config.get("settings", {})
        scan_paths = settings.get("scan_paths", {})
        
        image_path = scan_paths.get("image", self.base_dir)
        text_path = scan_paths.get("text", os.path.join(self.base_dir, "text"))
        cah_path = scan_paths.get("cah", os.path.join(self.base_dir, "cah"))
        
        # Scan main directory (categorize by detected use case)
        for lora in self.scan_directory(image_path):
            if lora.use_case == LoRAUseCase.IMAGE:
                self.image_loras[lora.name] = lora
            elif lora.use_case == LoRAUseCase.TEXT:
                self.text_loras[lora.name] = lora
            elif lora.use_case == LoRAUseCase.CAH:
                self.cah_loras[lora.name] = lora
        
        # Scan dedicated text directory
        for lora in self.scan_directory(text_path):
            lora.use_case = LoRAUseCase.TEXT
            self.text_loras[lora.name] = lora
        
        # Scan dedicated CAH directory
        for lora in self.scan_directory(cah_path):
            lora.use_case = LoRAUseCase.CAH
            self.cah_loras[lora.name] = lora
        
        logger.info(f"LoRA scan complete: {len(self.image_loras)} image, {len(self.text_loras)} text, {len(self.cah_loras)} CAH")
    
    def get_compatible_loras(self, use_case: LoRAUseCase, loaded_model_name: str = None, 
                              loaded_model_type: str = None) -> List[LoRAInfo]:
        """Get LoRAs compatible with the currently loaded model.
        
        Args:
            use_case: Filter by use case (image/text/cah)
            loaded_model_name: Name of currently loaded model (e.g., "Dreamshaper 8")
            loaded_model_type: Type/architecture (e.g., "sd15", "sdxl", "mistral")
        
        Returns:
            List of compatible LoRAInfo objects
        """
        # Get appropriate dict
        if use_case == LoRAUseCase.IMAGE:
            all_loras = self.image_loras
        elif use_case == LoRAUseCase.TEXT:
            all_loras = self.text_loras
        else:
            all_loras = self.cah_loras
        
        if not loaded_model_type:
            return list(all_loras.values())
        
        # Filter by base model compatibility
        compatible = []
        for lora in all_loras.values():
            # Check base model type match
            if lora.base_model.value == loaded_model_type.lower():
                compatible.append(lora)
            elif lora.base_model == LoRABaseModel.UNKNOWN:
                # Unknown base model - include but mark as potentially incompatible
                compatible.append(lora)
            
            # Check specific model compatibility if specified
            if lora.compatible_models and loaded_model_name:
                if loaded_model_name in lora.compatible_models:
                    if lora not in compatible:
                        compatible.append(lora)
        
        return compatible
    
    def get_lora_choices(self, use_case: LoRAUseCase, loaded_model_type: str = None) -> List[str]:
        """Get list of LoRA names for dropdown choices.
        
        Args:
            use_case: Filter by use case
            loaded_model_type: Optional model type for filtering
        
        Returns:
            List of LoRA names with "None" as first option
        """
        loras = self.get_compatible_loras(use_case, loaded_model_type=loaded_model_type)
        names = ["None"] + sorted([l.name for l in loras])
        return names
    
    def get_lora_info(self, name: str) -> Optional[LoRAInfo]:
        """Get LoRA info by name.
        
        Args:
            name: LoRA name
        
        Returns:
            LoRAInfo or None if not found
        """
        # Check all dicts
        if name in self.image_loras:
            return self.image_loras[name]
        if name in self.text_loras:
            return self.text_loras[name]
        if name in self.cah_loras:
            return self.cah_loras[name]
        return None
    
    def update_lora_metadata(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update LoRA metadata and save to disk.
        
        Args:
            name: LoRA name
            updates: Dict of fields to update
        
        Returns:
            True if successful
        """
        lora = self.get_lora_info(name)
        if not lora:
            return False
        
        # Update in-memory
        for key, value in updates.items():
            if hasattr(lora, key):
                setattr(lora, key, value)
        
        # Save to lora_metadata.json
        metadata_path = os.path.join(lora.path, "lora_metadata.json")
        try:
            # Load existing
            existing = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            
            # Update
            existing.update({
                "name": lora.name,
                "trigger_word": lora.trigger_words[0] if lora.trigger_words else "",
                "trigger_words": lora.trigger_words,
                "use_case": lora.use_case.value,
                "base_model": lora.base_model.value,
                "compatible_models": lora.compatible_models,
                "default_weight": lora.default_weight,
                "category": lora.category,
                "description": lora.description,
            })
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update LoRA metadata: {e}")
            return False
    
    def get_all_loras_summary(self) -> str:
        """Get summary of all discovered LoRAs for display."""
        lines = []
        
        if self.image_loras:
            lines.append(f"**Image LoRAs ({len(self.image_loras)}):**")
            for name, lora in sorted(self.image_loras.items()):
                base = lora.base_model.value.upper()
                trigger = f" | Trigger: {', '.join(lora.trigger_words)}" if lora.trigger_words else ""
                lines.append(f"  • {name} [{base}]{trigger}")
        
        if self.text_loras:
            lines.append(f"\n**Text LoRAs ({len(self.text_loras)}):**")
            for name, lora in sorted(self.text_loras.items()):
                base = lora.base_model.value.upper()
                lines.append(f"  • {name} [{base}]")
        
        if self.cah_loras:
            lines.append(f"\n**CAH LoRAs ({len(self.cah_loras)}):**")
            for name, lora in sorted(self.cah_loras.items()):
                base = lora.base_model.value.upper()
                lines.append(f"  • {name} [{base}]")
        
        if not any([self.image_loras, self.text_loras, self.cah_loras]):
            lines.append("*No LoRAs found. Place LoRA folders in `data/lora_models/`*")
        
        return "\n".join(lines)


# Singleton instance
_lora_manager: Optional[LoRAManager] = None


def get_lora_manager() -> LoRAManager:
    """Get or create the LoRA manager singleton."""
    global _lora_manager
    if _lora_manager is None:
        _lora_manager = LoRAManager()
    return _lora_manager
