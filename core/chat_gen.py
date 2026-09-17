"""
Core chat generation engine for KVGenius.
UI-agnostic - talks to a local Ollama server, does not load models in-process.

This is used only by the Card Generator (ui/tabs/card_generator.py) for
one-shot text generation - there is no conversation history, persona system,
or memory extraction here; those were part of the Chat tab, which has been
removed (chat now lives in a separate app).
"""
import logging
from typing import Optional, Dict, Any, Callable, Tuple, List

from . import ollama_client
from .config import CHAT_MODELS

logger = logging.getLogger(__name__)

# Global state
_current_chat_model_key: Optional[str] = None  # display key from CHAT_MODELS
_current_chat_model_id: Optional[str] = None    # Ollama model tag

# Debug info from the last generation call — read by the Card Generator's debug view
_last_debug_info: Dict[str, Any] = {}


def get_available_chat_models() -> Dict[str, Dict[str, Any]]:
    """Get all available chat models from config. 'id' is the Ollama model tag."""
    models = {}

    for name, preset in CHAT_MODELS.items():
        models[name] = {
            "id": preset.get("id", ""),
            "description": preset.get("description", ""),
            "details": preset.get("details", ""),
            "best_for": preset.get("best_for", ""),
            "params": preset.get("params", ""),
            "vram": preset.get("vram", ""),
        }

    return models


def is_chat_model_downloaded(repo_id: str) -> bool:
    """Check if a model tag is already pulled in Ollama."""
    try:
        return ollama_client.is_model_available(repo_id)
    except ollama_client.OllamaUnavailableError:
        return False


def _is_chat_model_fully_downloaded(repo_id: str) -> bool:
    """Alias for is_chat_model_downloaded - Ollama doesn't expose partial-pull state."""
    return is_chat_model_downloaded(repo_id)


def unload_chat_model():
    """Ask Ollama to unload the current chat model from VRAM."""
    global _current_chat_model_key, _current_chat_model_id

    if _current_chat_model_id is not None:
        logger.info(f"Unloading chat model: {_current_chat_model_key}")
        ollama_client.unload_model(_current_chat_model_id)

    _current_chat_model_key = None
    _current_chat_model_id = None


def load_chat_model(
    model_key: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Point KVGenius at an Ollama model. Does not load weights itself - Ollama
    manages that server-side; this just validates the tag is available.

    Args:
        model_key: Name of the model preset to load
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Tuple of (success: bool, message: str)
    """
    global _current_chat_model_key, _current_chat_model_id

    def report_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct*100:.0f}%] {msg}")

    models = get_available_chat_models()
    if model_key not in models:
        return False, f"Unknown model: {model_key}"

    model_id = models[model_key]["id"]

    if _current_chat_model_key == model_key and _current_chat_model_id is not None:
        return True, f"{model_key} already loaded"

    report_progress(0.2, f"Checking Ollama for {model_id}...")

    try:
        if not ollama_client.is_model_available(model_id):
            return False, (
                f"'{model_id}' is not pulled in Ollama yet. "
                f"Download it first (or run: ollama pull {model_id})."
            )
    except ollama_client.OllamaUnavailableError as e:
        return False, str(e)

    _current_chat_model_key = model_key
    _current_chat_model_id = model_id

    report_progress(1.0, f"{model_key} ready!")
    return True, f"Successfully loaded {model_key}"


def _render_messages_for_debug(messages: List[Dict[str, str]]) -> str:
    """Render a messages list as readable text for the debug view."""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def generate_chat_response(
    user_message: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    extra_stop_phrases: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Generate a one-shot chat response via Ollama (no conversation history).

    Args:
        user_message: The prompt to send
        system_prompt: Optional system prompt for AI behavior
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        extra_stop_phrases: Optional list of additional stop phrases to truncate the response

    Returns:
        Tuple of (success: bool, response: str)
    """
    if _current_chat_model_id is None:
        return False, "No chat model loaded. Please load a model first."

    try:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        stop_phrases = list(extra_stop_phrases) if extra_stop_phrases else []

        # Capture debug info
        _last_debug_info.clear()
        _last_debug_info["system_prompt"] = system_prompt or ""
        _last_debug_info["user_message"] = user_message
        _last_debug_info["full_prompt"] = _render_messages_for_debug(messages)
        _last_debug_info["stop_phrases"] = stop_phrases

        options = {
            "temperature": max(temperature, 0.1),
            "top_p": min(max(top_p, 0.1), 1.0),
            "top_k": top_k if top_k > 0 else 50,
            "repeat_penalty": max(repetition_penalty, 1.0),
            "num_predict": max_new_tokens,
        }
        if stop_phrases:
            options["stop"] = stop_phrases

        result = ollama_client.chat(_current_chat_model_id, messages, options=options)
        response = (result.get("message") or {}).get("content", "").strip()

        _last_debug_info["raw_response"] = response
        _last_debug_info["input_tokens"] = result.get("prompt_eval_count", "?")
        _last_debug_info["output_tokens"] = result.get("eval_count", "?")
        _last_debug_info["cleaned_response"] = response

        return True, response

    except ollama_client.OllamaUnavailableError as e:
        error_msg = str(e)
        _last_debug_info["error"] = error_msg
        logger.error(f"Generation error: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = str(e)
        _last_debug_info["error"] = error_msg
        logger.error(f"Generation error: {error_msg}")
        return False, f"Error: {error_msg}"


def download_chat_model(
    model_tag: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """
    Pull a chat model into Ollama.

    Args:
        model_tag: Ollama model tag (e.g. "mistral:7b")
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Tuple of (success: bool, message: str)
    """
    if is_chat_model_downloaded(model_tag):
        return True, f"{model_tag} is already downloaded"

    try:
        return ollama_client.pull(model_tag, progress_callback=progress_callback)
    except ollama_client.OllamaUnavailableError as e:
        return False, str(e)
