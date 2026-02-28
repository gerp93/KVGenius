"""
Core chat generation engine for KVGenius.
UI-agnostic - can be used by Gradio, Flet, or CLI.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass, field

# Ensure fix_dll_paths is imported before torch
sys.path.insert(0, str(Path(__file__).parent.parent))
import fix_dll_paths

import torch

from .config import CACHE_DIR, CHAT_MODELS

logger = logging.getLogger(__name__)

# Global state
_current_chat_model: Optional[Any] = None
_current_chat_model_key: Optional[str] = None
_current_tokenizer: Optional[Any] = None
_conversation_history: List[Dict[str, str]] = []
_current_lora: Optional[str] = None  # Currently loaded LoRA name
_base_model_backup: Optional[Any] = None  # Backup of base model before LoRA


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResult:
    """Result of chat generation."""
    response: str
    tokens_generated: int
    generation_time: float


def get_available_chat_models() -> Dict[str, Dict[str, Any]]:
    """Get all available chat models from config."""
    models = {}
    
    for name, preset in CHAT_MODELS.items():
        models[name] = {
            "id": preset.get("id", ""),
            "description": preset.get("description", ""),
            "details": preset.get("details", ""),
            "best_for": preset.get("best_for", ""),
            "params": preset.get("params", ""),
            "vram": preset.get("vram", "~14 GB"),
        }
    
    return models


def is_chat_model_downloaded(repo_id: str) -> bool:
    """Check if a chat model is downloaded."""
    cache_name = "models--" + repo_id.replace("/", "--")
    cache_path = CACHE_DIR / cache_name
    
    if not cache_path.exists():
        return False
    
    # Check for model files (>100MB)
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


def _is_chat_model_fully_downloaded(repo_id: str) -> bool:
    """Check if a chat model is fully downloaded.
    
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
    
    # Verify refs/ and snapshots/ exist
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


def get_current_chat_model() -> Tuple[Optional[str], bool]:
    """Get the currently loaded chat model key and loaded status."""
    return _current_chat_model_key, _current_chat_model is not None


def get_current_lora() -> Optional[str]:
    """Get the currently loaded LoRA name."""
    return _current_lora


def unload_chat_model():
    """Unload current chat model to free VRAM."""
    global _current_chat_model, _current_chat_model_key, _current_tokenizer, _current_lora
    
    # Unload LoRA first if loaded
    if _current_lora:
        unload_chat_lora()
    
    if _current_chat_model is not None:
        logger.info(f"Unloading chat model: {_current_chat_model_key}")
        del _current_chat_model
        del _current_tokenizer
        _current_chat_model = None
        _current_chat_model_key = None
        _current_tokenizer = None
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_chat_lora(lora_name: str, lora_type: str = "cardgen") -> Tuple[bool, str]:
    """
    Load a LoRA adapter onto the current chat model.
    
    Args:
        lora_name: Name of the LoRA to load
        lora_type: Type of LoRA ("cardgen" for card generation, "text" for personas)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    global _current_chat_model, _current_lora, _current_chat_model_key
    
    if _current_chat_model is None:
        return False, "No chat model loaded"
    
    # Determine LoRA directory
    if lora_type == "cardgen":
        lora_dir = Path("./data/lora_models/cardgen")
    else:
        lora_dir = Path("./data/lora_models/text")
    
    lora_path = lora_dir / lora_name
    
    if not lora_path.exists():
        return False, f"LoRA not found: {lora_name}"
    
    # Check for adapter files
    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return False, f"Invalid LoRA (no adapter_config.json): {lora_name}"
    
    # Load and check adapter config for compatibility
    try:
        import json
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        
        base_model = adapter_config.get("base_model_name_or_path", "")
        
        # Check if current model matches the LoRA's base model
        current_model_id = CHAT_MODELS.get(_current_chat_model_key, {}).get("model_id", "")
        if base_model and current_model_id:
            # Normalize model names for comparison
            base_model_norm = base_model.lower().replace("-", "").replace("_", "")
            current_model_norm = current_model_id.lower().replace("-", "").replace("_", "")
            
            if base_model_norm not in current_model_norm and current_model_norm not in base_model_norm:
                return False, f"LoRA '{lora_name}' was trained on '{base_model}' but current model is '{current_model_id}'. Please load the correct model."
    except Exception as e:
        logger.warning(f"Could not verify LoRA compatibility: {e}")
    
    try:
        from peft import PeftModel
        
        # Unload existing LoRA if different
        if _current_lora and _current_lora != lora_name:
            unload_chat_lora()
        
        if _current_lora == lora_name:
            return True, f"LoRA {lora_name} already loaded"
        
        logger.info(f"Loading LoRA: {lora_name} from {lora_path}")
        
        # Load LoRA adapter with explicit settings to avoid dtype issues
        _current_chat_model = PeftModel.from_pretrained(
            _current_chat_model,
            str(lora_path),
            is_trainable=False,
            torch_dtype=_current_chat_model.dtype if hasattr(_current_chat_model, 'dtype') else torch.float16,
        )
        
        # Ensure model is in eval mode
        _current_chat_model.eval()
        
        _current_lora = lora_name
        logger.info(f"LoRA loaded successfully: {lora_name}")
        
        return True, f"LoRA {lora_name} loaded"
        
    except Exception as e:
        logger.error(f"Failed to load LoRA {lora_name}: {e}")
        return False, f"Failed to load LoRA: {str(e)}"


def unload_chat_lora() -> Tuple[bool, str]:
    """
    Unload the currently loaded LoRA and restore base model.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    global _current_chat_model, _current_lora
    
    if _current_lora is None:
        return True, "No LoRA loaded"
    
    try:
        # Merge and unload - this creates a clean base model
        if hasattr(_current_chat_model, 'merge_and_unload'):
            _current_chat_model = _current_chat_model.merge_and_unload()
            logger.info(f"Unloaded LoRA: {_current_lora}")
        elif hasattr(_current_chat_model, 'unload'):
            _current_chat_model.unload()
            logger.info(f"Unloaded LoRA: {_current_lora}")
        
        old_lora = _current_lora
        _current_lora = None
        
        return True, f"LoRA {old_lora} unloaded"
        
    except Exception as e:
        logger.error(f"Failed to unload LoRA: {e}")
        return False, f"Failed to unload LoRA: {str(e)}"


def get_available_chat_loras(lora_type: str = "cardgen") -> List[Dict[str, Any]]:
    """
    Get list of available chat LoRAs.
    
    Args:
        lora_type: "cardgen" for card generation LoRAs, "text" for persona LoRAs
    
    Returns:
        List of LoRA info dicts
    """
    import json
    
    if lora_type == "cardgen":
        lora_dir = Path("./data/lora_models/cardgen")
    else:
        lora_dir = Path("./data/lora_models/text")
    
    if not lora_dir.exists():
        return []
    
    loras = []
    for folder in lora_dir.iterdir():
        if not folder.is_dir():
            continue
        
        adapter_config = folder / "adapter_config.json"
        if not adapter_config.exists():
            continue
        
        # Load metadata if available
        metadata_path = folder / "training_metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except:
                pass
        
        loras.append({
            "name": folder.name,
            "description": metadata.get("description", ""),
            "base_model": metadata.get("base_model", "Unknown"),
            "created_at": metadata.get("created_at", ""),
        })
    
    return sorted(loras, key=lambda x: x["name"])


def load_chat_model(
    model_key: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Load a chat model.
    
    Args:
        model_key: Name of the model to load
        progress_callback: Optional callback(progress: 0-1, message: str)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    global _current_chat_model, _current_chat_model_key, _current_tokenizer
    
    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct*100:.0f}%] {msg}")
    
    models = get_available_chat_models()
    if model_key not in models:
        return False, f"Unknown model: {model_key}"
    
    model_info = models[model_key]
    model_id = model_info["id"]
    
    # Check if already loaded
    if _current_chat_model_key == model_key and _current_chat_model is not None:
        return True, f"{model_key} already loaded"
    
    # Unload previous model
    if _current_chat_model is not None:
        report_progress(0.05, "Unloading previous model...")
        unload_chat_model()
    
    report_progress(0.1, f"Loading {model_key}...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        report_progress(0.2, "Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        report_progress(0.4, "Loading model weights...")
        
        # Determine dtype for compute
        dtype = torch.float16
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
        
        # Always use 4-bit quantization for chat models
        # Benefits: ~4x less VRAM, compatible with QLoRA-trained LoRAs, fast inference
        logger.info("Loading model with 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(CACHE_DIR),
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        report_progress(0.9, "Finalizing...")
        
        _current_chat_model = model
        _current_tokenizer = tokenizer
        _current_chat_model_key = model_key
        
        report_progress(1.0, f"{model_key} loaded!")
        return True, f"Successfully loaded {model_key}"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error loading chat model: {error_msg}")
        return False, f"Failed to load: {error_msg}"


def clear_conversation():
    """Clear the conversation history."""
    global _conversation_history
    _conversation_history = []


def get_conversation_history() -> List[Dict[str, str]]:
    """Get the current conversation history."""
    return _conversation_history.copy()


def format_prompt(
    user_message: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Format prompt for the current model."""
    global _current_tokenizer, _current_chat_model_key
    
    if _current_tokenizer is None:
        return user_message
    
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add history
    if history:
        messages.extend(history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Try using tokenizer's chat template
    if hasattr(_current_tokenizer, 'apply_chat_template') and _current_tokenizer.chat_template:
        try:
            return _current_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")
    
    # Fallback to model-specific formats
    model_key = _current_chat_model_key or ""
    
    # Mistral/Dolphin format
    if "Mistral" in model_key or "Dolphin" in model_key or "Kunoichi" in model_key:
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"[INST] {system_prompt} [/INST] Understood.</s>")
        
        if history:
            for msg in history:
                if msg["role"] == "user":
                    prompt_parts.append(f"[INST] {msg['content']} [/INST]")
                elif msg["role"] == "assistant":
                    prompt_parts[-1] += f" {msg['content']}</s>"
        
        prompt_parts.append(f"[INST] {user_message} [/INST]")
        return "".join(prompt_parts)
    
    # Llama/ChatML format
    if "llama" in model_key.lower() or "Llama" in model_key:
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        
        if history:
            for msg in history:
                prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        
        prompt_parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    # Generic fallback
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    if history:
        for msg in history:
            role = msg["role"].capitalize()
            prompt_parts.append(f"{role}: {msg['content']}")
    prompt_parts.append(f"User: {user_message}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def generate_chat_response(
    user_message: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_history: bool = True,
) -> Tuple[bool, str]:
    """
    Generate a chat response.
    
    Args:
        user_message: The user's message
        system_prompt: Optional system prompt for AI behavior
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        progress_callback: Optional callback for streaming tokens
        use_history: Whether to use/update conversation history (default True)
    
    Returns:
        Tuple of (success: bool, response: str)
    """
    global _current_chat_model, _current_tokenizer, _conversation_history
    
    if _current_chat_model is None or _current_tokenizer is None:
        return False, "No chat model loaded. Please load a model first."
    
    try:
        # Format prompt - only include history if use_history is True
        history_to_use = _conversation_history if use_history else []
        prompt = format_prompt(user_message, system_prompt, history_to_use)
        
        # Tokenize
        inputs = _current_tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(_current_chat_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = _current_chat_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.1),
                top_p=min(max(top_p, 0.1), 1.0),
                top_k=top_k if top_k > 0 else 50,
                repetition_penalty=max(repetition_penalty, 1.0),
                do_sample=True,
                pad_token_id=_current_tokenizer.pad_token_id,
                eos_token_id=_current_tokenizer.eos_token_id,
            )
        
        # Decode response
        response = _current_tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        
        # Clean up response
        stop_phrases = ["\nUser:", "\nAssistant:", "\n\n\n", "<|im_end|>", "</s>", "[INST]"]
        for stop in stop_phrases:
            if stop in response:
                response = response.split(stop)[0].strip()
        
        # Only add to history if use_history is True
        if use_history:
            _conversation_history.append({"role": "user", "content": user_message})
            _conversation_history.append({"role": "assistant", "content": response})
            
            # Keep history manageable (last 10 exchanges)
            if len(_conversation_history) > 20:
                _conversation_history = _conversation_history[-20:]
        
        return True, response
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Generation error: {error_msg}")
        return False, f"Error: {error_msg}"


def extract_memories(
    user_message: str,
    ai_response: str,
    existing_memories: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """
    Ask the loaded model to extract narrative memories from the latest exchange.
    
    Focuses on events, promises, emotional moments, and relationship developments
    that happen DURING conversation — NOT static character traits or descriptions
    already present in the system prompt.
    
    Returns a list of short memory strings (may be empty if nothing notable).
    """
    global _current_chat_model, _current_tokenizer
    
    if _current_chat_model is None or _current_tokenizer is None:
        return []
    
    existing_str = ""
    if existing_memories:
        existing_str = "\n".join(f"- {m}" for m in existing_memories)
        existing_str = f"\nAlready recorded memories (do NOT repeat these):\n{existing_str}\n"
    
    system_note = ""
    if system_prompt:
        # Truncate to avoid blowing up the extraction prompt
        sp_trimmed = system_prompt[:600] + ("..." if len(system_prompt) > 600 else "")
        system_note = f"\nThe following character/setting info is ALREADY in the system prompt (do NOT record any of this):\n{sp_trimmed}\n"
    
    extraction_prompt = f"""You are a memory extraction assistant. Read the conversation exchange below and identify ONLY narrative events worth remembering for future conversations.

RECORD things like:
- Promises or commitments made (e.g., "User promised to visit tomorrow")
- Emotional moments (e.g., "Character was sad when User mentioned leaving")
- Actions that happened (e.g., "They shared a meal together")
- New information User revealed about themselves (e.g., "User mentioned they have a sister")
- Relationship changes (e.g., "User and Character had their first argument")
- Plans or decisions made (e.g., "They agreed to meet at the park")

DO NOT record:
- Character personality traits, appearance, or backstory (already in system prompt)
- General facts about the setting or world
- Things that are just conversational filler or greetings
- Anything already in the recorded memories list

If nothing notable happened in this exchange, output ONLY "NONE".
Otherwise output one memory per line, prefixed with "- ". Keep each memory brief (one sentence).
{system_note}{existing_str}
User: {user_message}
Assistant: {ai_response}

Notable events from this exchange:"""
    
    try:
        inputs = _current_tokenizer.encode(
            extraction_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(_current_chat_model.device)
        
        with torch.no_grad():
            outputs = _current_chat_model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                pad_token_id=_current_tokenizer.pad_token_id,
                eos_token_id=_current_tokenizer.eos_token_id,
            )
        
        result = _current_tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        
        # Clean up stop tokens
        stop_phrases = ["\nUser:", "\nAssistant:", "\n\n\n", "<|im_end|>", "</s>", "[INST]"]
        for stop in stop_phrases:
            if stop in result:
                result = result.split(stop)[0].strip()
        
        if "NONE" in result.upper() and len(result) < 20:
            return []
        
        # Parse bullet points
        memories = []
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                fact = line[2:].strip()
                if fact and len(fact) > 5 and len(fact) < 300:
                    memories.append(fact)
        
        # Post-filter: reject memories that are just restating system prompt info
        if system_prompt and memories:
            memories = _filter_redundant_memories(memories, system_prompt, existing_memories)
        
        logger.info(f"Extracted {len(memories)} memories from exchange")
        return memories
        
    except Exception as e:
        logger.warning(f"Memory extraction failed (non-critical): {e}")
        return []


def _filter_redundant_memories(
    memories: List[str],
    system_prompt: str,
    existing_memories: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter out extracted memories that are redundant with the system prompt
    or already-existing memories. Uses simple substring/keyword overlap.
    """
    sp_lower = system_prompt.lower()
    existing_lower = set()
    if existing_memories:
        existing_lower = {m.lower().strip() for m in existing_memories}
    
    filtered = []
    for mem in memories:
        mem_lower = mem.lower().strip()
        
        # Skip if it's a near-duplicate of an existing memory
        if any(_text_similarity(mem_lower, ex) > 0.65 for ex in existing_lower):
            logger.debug(f"Filtered duplicate memory: {mem}")
            continue
        
        # Skip if the core content is already in the system prompt
        # Extract meaningful words (skip short/common words)
        words = [w for w in mem_lower.split() if len(w) > 3]
        if len(words) > 2:
            # If >60% of the meaningful words are in the system prompt, skip
            overlap = sum(1 for w in words if w in sp_lower)
            overlap_ratio = overlap / len(words)
            if overlap_ratio > 0.6:
                logger.debug(f"Filtered system-prompt-redundant memory ({overlap_ratio:.0%}): {mem}")
                continue
        
        # Skip very generic/vague memories
        generic_phrases = [
            "is a character", "is described as", "has a personality",
            "the setting is", "takes place in", "is known for",
            "character traits", "appearance includes",
        ]
        if any(gp in mem_lower for gp in generic_phrases):
            logger.debug(f"Filtered generic memory: {mem}")
            continue
        
        filtered.append(mem)
    
    return filtered


def _text_similarity(a: str, b: str) -> float:
    """Simple word-overlap Jaccard similarity between two strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def download_chat_model(
    repo_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Download a chat model from HuggingFace.
    
    Args:
        repo_id: HuggingFace repository ID
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
    
    if _is_chat_model_fully_downloaded(repo_id):
        return True, f"{repo_id} is already downloaded"
    
    report_progress(0.02, f"Starting download: {repo_id}")
    
    stop_monitor = threading.Event()
    
    try:
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
            max_workers=1,
            resume_download=True,
        )
        
        stop_monitor.set()
        
        report_progress(1.0, f"Download complete: {repo_id}")
        return True, f"Successfully downloaded {repo_id}"
        
    except Exception as e:
        stop_monitor.set()
        error_msg = str(e)
        logger.error(f"Download error: {error_msg}")
        return False, f"Download failed: {error_msg}"
