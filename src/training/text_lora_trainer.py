"""
Text LoRA Trainer Module for KVGenius
Handles training LoRAs for text/chat models using PEFT.
Used by both Text LoRA Training and Card LoRA Training tabs.
"""

import os
import json
import shutil
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)

# Directories - all user data goes in data/
TEXT_LORA_DIR = Path("./data/lora_models/text")
CARDGEN_LORA_DIR = Path("./data/lora_models/cardgen")
MODEL_CACHE_DIR = Path("./data/model_cache")

# Ensure directories exist
TEXT_LORA_DIR.mkdir(parents=True, exist_ok=True)
CARDGEN_LORA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TextLoRAType(Enum):
    """Type of text LoRA being trained."""
    PERSONA = "persona"         # Character/persona training
    CHAT_STYLE = "chat_style"   # Chat style training
    KNOWLEDGE = "knowledge"     # Knowledge domain training
    CARD_GENERATOR = "cardgen"  # CAH card generation training


@dataclass
class TextTrainingExample:
    """A single training example for text LoRA."""
    input_text: str       # Input/prompt text
    output_text: str      # Expected output/response
    instruction: str = "" # Optional instruction prefix


@dataclass
class TextLoRAInfo:
    """Information about a trained text LoRA."""
    name: str
    lora_type: str  # TextLoRAType value
    base_model: str
    created_at: str
    training_steps: int
    rank: int
    alpha: float
    description: str = ""
    trigger_word: str = ""
    file_path: Optional[str] = None
    file_size_mb: float = 0.0
    # Training metadata
    num_examples: int = 0
    final_loss: float = 0.0


@dataclass
class TextTrainingConfig:
    """Configuration for text LoRA training."""
    output_name: str
    lora_type: str  # TextLoRAType value
    description: str = ""
    
    # Base model - uses currently loaded chat model if not specified
    base_model: Optional[str] = None
    
    # Training data
    training_examples: List[TextTrainingExample] = field(default_factory=list)
    
    # Training params
    num_train_epochs: int = 3
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    
    # LoRA params
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Memory optimization
    use_8bit: bool = True
    gradient_checkpointing: bool = True
    max_seq_length: int = 512
    
    # Card-specific options (for card generator training)
    card_format: str = "instruction"  # "instruction" or "completion"


class TextTrainingState:
    """Holds the current training state for UI updates."""
    
    def __init__(self):
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss = 0.0
        self.status_message = "Idle"
        self.error: Optional[str] = None
        self.completed = False
        self.output_path: Optional[str] = None
        self.progress_pct = 0.0


# Global training state
text_training_state = TextTrainingState()


class TextLoRAManager:
    """Manages trained text LoRA files."""
    
    def __init__(self, lora_type: TextLoRAType = TextLoRAType.PERSONA):
        self.lora_type = lora_type
        self.lora_dir = CARDGEN_LORA_DIR if lora_type == TextLoRAType.CARD_GENERATOR else TEXT_LORA_DIR
        self.lora_dir.mkdir(parents=True, exist_ok=True)
    
    def list_loras(self) -> List[TextLoRAInfo]:
        """List all available text LoRAs of this type."""
        loras = []
        
        for lora_folder in self.lora_dir.iterdir():
            if not lora_folder.is_dir():
                continue
            
            # Look for adapter_config.json (PEFT format)
            config_path = lora_folder / "adapter_config.json"
            if not config_path.exists():
                continue
            
            name = lora_folder.name
            
            # Load metadata
            metadata_path = lora_folder / "training_metadata.json"
            info = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        info = json.load(f)
                except:
                    pass
            
            # Get file size
            adapter_path = lora_folder / "adapter_model.safetensors"
            file_size = adapter_path.stat().st_size / (1024 * 1024) if adapter_path.exists() else 0
            
            loras.append(TextLoRAInfo(
                name=name,
                lora_type=info.get("lora_type", self.lora_type.value),
                base_model=info.get("base_model", "Unknown"),
                created_at=info.get("created_at", "Unknown"),
                training_steps=info.get("training_steps", 0),
                rank=info.get("rank", 16),
                alpha=info.get("alpha", 32),
                description=info.get("description", ""),
                trigger_word=info.get("trigger_word", ""),
                file_path=str(lora_folder),
                file_size_mb=file_size,
                num_examples=info.get("num_examples", 0),
                final_loss=info.get("final_loss", 0.0)
            ))
        
        return sorted(loras, key=lambda x: x.name)
    
    def get_lora(self, name: str) -> Optional[TextLoRAInfo]:
        """Get info for a specific LoRA."""
        lora_folder = self.lora_dir / name
        config_path = lora_folder / "adapter_config.json"
        
        if not config_path.exists():
            return None
        
        metadata_path = lora_folder / "training_metadata.json"
        info = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    info = json.load(f)
            except:
                pass
        
        adapter_path = lora_folder / "adapter_model.safetensors"
        file_size = adapter_path.stat().st_size / (1024 * 1024) if adapter_path.exists() else 0
        
        return TextLoRAInfo(
            name=name,
            lora_type=info.get("lora_type", self.lora_type.value),
            base_model=info.get("base_model", "Unknown"),
            created_at=info.get("created_at", "Unknown"),
            training_steps=info.get("training_steps", 0),
            rank=info.get("rank", 16),
            alpha=info.get("alpha", 32),
            description=info.get("description", ""),
            trigger_word=info.get("trigger_word", ""),
            file_path=str(lora_folder),
            file_size_mb=file_size,
            num_examples=info.get("num_examples", 0),
            final_loss=info.get("final_loss", 0.0)
        )
    
    def delete_lora(self, name: str) -> bool:
        """Delete a LoRA and its files."""
        lora_folder = self.lora_dir / name
        if lora_folder.exists():
            shutil.rmtree(lora_folder)
            return True
        return False


class TextLoRATrainer:
    """Handles text LoRA training with background execution."""
    
    def __init__(self, lora_type: TextLoRAType = TextLoRAType.PERSONA):
        self.lora_type = lora_type
        self.lora_dir = CARDGEN_LORA_DIR if lora_type == TextLoRAType.CARD_GENERATOR else TEXT_LORA_DIR
        self._training_thread: Optional[threading.Thread] = None
        self._stop_requested = False
    
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return text_training_state.is_training
    
    def get_state(self) -> TextTrainingState:
        """Get current training state."""
        return text_training_state
    
    def request_stop(self):
        """Request training to stop."""
        self._stop_requested = True
        text_training_state.status_message = "Stopping..."
    
    def validate_config(self, config: TextTrainingConfig) -> tuple[bool, str]:
        """Validate training configuration before starting."""
        if not config.output_name or not config.output_name.strip():
            return False, "Output name is required"
        
        if not config.training_examples or len(config.training_examples) < 3:
            return False, "Need at least 3 training examples"
        
        if config.lora_rank < 4 or config.lora_rank > 256:
            return False, "LoRA rank must be between 4 and 256"
        
        return True, ""
    
    def start_training(self, config: TextTrainingConfig,
                       progress_callback: Optional[Callable] = None) -> bool:
        """Start training in background thread."""
        global text_training_state
        
        if text_training_state.is_training:
            return False
        
        # Validate config
        valid, error = self.validate_config(config)
        if not valid:
            text_training_state.error = error
            text_training_state.status_message = f"Error: {error}"
            return False
        
        self._stop_requested = False
        text_training_state = TextTrainingState()
        text_training_state.is_training = True
        text_training_state.status_message = "Initializing..."
        
        def train_thread():
            try:
                self._run_training(config, progress_callback)
            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                text_training_state.error = str(e)
                text_training_state.status_message = f"Error: {e}"
            finally:
                text_training_state.is_training = False
        
        self._training_thread = threading.Thread(target=train_thread, daemon=True)
        self._training_thread.start()
        
        return True
    
    def _run_training(self, config: TextTrainingConfig,
                      progress_callback: Optional[Callable] = None):
        """Run the actual training loop."""
        global text_training_state
        
        text_training_state.status_message = "Checking dependencies..."
        
        # Check for required packages
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling
            )
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            from datasets import Dataset
        except ImportError as e:
            text_training_state.error = f"Missing required packages: {e}. Install with: pip install peft transformers datasets"
            text_training_state.status_message = text_training_state.error
            return
        
        # Determine base model
        base_model_id = config.base_model
        if not base_model_id:
            # Try to get from app state (loaded chat model)
            try:
                from ui.state import app_state
                from core.chat_gen import get_current_chat_model, CHAT_MODELS
                from core.config import CHAT_MODELS as CHAT_MODEL_CONFIG
                
                model_key, is_loaded = get_current_chat_model()
                if model_key and is_loaded:
                    base_model_id = CHAT_MODEL_CONFIG.get(model_key, {}).get("id")
            except:
                pass
        
        if not base_model_id:
            text_training_state.error = "No base model specified and no chat model is loaded. Load a chat model first or specify a base model."
            text_training_state.status_message = text_training_state.error
            return
        
        text_training_state.status_message = f"Loading base model: {base_model_id}..."
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                cache_dir=str(MODEL_CACHE_DIR),
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization if requested
            model_kwargs = {
                "cache_dir": str(MODEL_CACHE_DIR),
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
            
            if config.use_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, training without 8-bit quantization")
            
            model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
            
            # Prepare for k-bit training if using quantization
            if config.use_8bit:
                model = prepare_model_for_kbit_training(model)
            
            # Enable gradient checkpointing
            if config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            text_training_state.status_message = "Configuring LoRA..."
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none"
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            text_training_state.status_message = "Preparing training data..."
            
            # Prepare training data
            def format_example(example: TextTrainingExample) -> str:
                """Format a training example as text."""
                if example.instruction:
                    return f"### Instruction:\n{example.instruction}\n\n### Input:\n{example.input_text}\n\n### Response:\n{example.output_text}"
                else:
                    return f"### Input:\n{example.input_text}\n\n### Response:\n{example.output_text}"
            
            # Tokenize examples
            formatted_texts = [format_example(ex) for ex in config.training_examples]
            
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_seq_length,
                    return_tensors="pt"
                )
            
            # Create dataset
            dataset = Dataset.from_dict({"text": formatted_texts})
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )
            
            # Calculate training steps
            num_examples = len(config.training_examples)
            steps_per_epoch = max(1, num_examples // (config.train_batch_size * config.gradient_accumulation_steps))
            total_steps = steps_per_epoch * config.num_train_epochs
            
            text_training_state.total_steps = total_steps
            text_training_state.total_epochs = config.num_train_epochs
            
            # Output directory
            output_dir = self.lora_dir / config.output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            text_training_state.status_message = "Starting training..."
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler,
                warmup_ratio=config.warmup_ratio,
                logging_steps=1,
                save_steps=steps_per_epoch,
                save_total_limit=2,
                fp16=True,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            
            # Custom callback for progress updates
            class ProgressCallback:
                def __init__(self, state: TextTrainingState, callback: Optional[Callable]):
                    self.state = state
                    self.callback = callback
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        self.state.loss = logs.get("loss", 0.0)
                        self.state.current_step = state.global_step
                        self.state.current_epoch = int(state.epoch) if state.epoch else 0
                        self.state.progress_pct = (state.global_step / self.state.total_steps) * 100
                        self.state.status_message = f"Epoch {self.state.current_epoch}/{self.state.total_epochs} | Step {state.global_step}/{self.state.total_steps} | Loss: {self.state.loss:.4f}"
                        
                        if self.callback:
                            self.callback(self.state)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            # Add callback
            progress_cb = ProgressCallback(text_training_state, progress_callback)
            trainer.add_callback(progress_cb)
            
            # Train!
            trainer.train()
            
            if self._stop_requested:
                text_training_state.status_message = "Training stopped by user"
                return
            
            text_training_state.status_message = "Saving LoRA..."
            
            # Save the model
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            # Save training metadata
            metadata = {
                "lora_type": config.lora_type,
                "base_model": base_model_id,
                "created_at": datetime.now().isoformat(),
                "training_steps": text_training_state.current_step,
                "rank": config.lora_rank,
                "alpha": config.lora_alpha,
                "description": config.description,
                "num_examples": num_examples,
                "final_loss": text_training_state.loss,
                "target_modules": config.target_modules
            }
            
            with open(output_dir / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            text_training_state.completed = True
            text_training_state.output_path = str(output_dir)
            text_training_state.status_message = f"✅ Training complete! LoRA saved to {config.output_name}"
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            text_training_state.error = str(e)
            text_training_state.status_message = f"Training failed: {e}"
            raise


# Helper functions for creating training examples

def create_persona_examples(persona_name: str, 
                           conversations: List[Dict[str, str]],
                           persona_description: str = "") -> List[TextTrainingExample]:
    """
    Create training examples for persona/character training.
    
    Args:
        persona_name: Name of the character/persona
        conversations: List of dicts with 'user' and 'assistant' keys
        persona_description: Optional description of the persona
    
    Returns:
        List of TextTrainingExample objects
    """
    examples = []
    instruction = f"You are {persona_name}. {persona_description}" if persona_description else f"You are {persona_name}."
    
    for conv in conversations:
        examples.append(TextTrainingExample(
            instruction=instruction,
            input_text=conv.get("user", ""),
            output_text=conv.get("assistant", "")
        ))
    
    return examples


def create_card_generation_examples(black_cards: List[str],
                                    white_cards: List[str],
                                    theme_name: str = "") -> List[TextTrainingExample]:
    """
    Create training examples for CAH card generation.
    
    Args:
        black_cards: List of black card texts (prompts with blanks)
        white_cards: List of white card texts (responses)
        theme_name: Optional theme name
    
    Returns:
        List of TextTrainingExample objects
    """
    examples = []
    theme_str = f" in the style of {theme_name}" if theme_name else ""
    
    # Black card examples
    for card in black_cards:
        examples.append(TextTrainingExample(
            instruction=f"Generate a Cards Against Humanity black card (prompt with blanks){theme_str}.",
            input_text="Create a funny black card",
            output_text=card
        ))
    
    # White card examples
    for card in white_cards:
        examples.append(TextTrainingExample(
            instruction=f"Generate a Cards Against Humanity white card (answer/response){theme_str}.",
            input_text="Create a funny white card",
            output_text=card
        ))
    
    return examples


# Global instances
def get_text_lora_trainer(lora_type: TextLoRAType = TextLoRAType.PERSONA) -> TextLoRATrainer:
    """Get a text LoRA trainer instance."""
    return TextLoRATrainer(lora_type)


def get_text_lora_manager(lora_type: TextLoRAType = TextLoRAType.PERSONA) -> TextLoRAManager:
    """Get a text LoRA manager instance."""
    return TextLoRAManager(lora_type)


def get_training_state() -> TextTrainingState:
    """Get the current training state."""
    return text_training_state
