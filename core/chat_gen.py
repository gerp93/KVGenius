"""
Core chat generation engine for KVGenius.
UI-agnostic - talks to a local Ollama server, does not load models in-process.
"""
import re
import logging
from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass

from . import ollama_client
from .config import CHAT_MODELS

logger = logging.getLogger(__name__)

# Global state
_current_chat_model_key: Optional[str] = None  # display key from CHAT_MODELS
_current_chat_model_id: Optional[str] = None    # Ollama model tag
_conversation_history: List[Dict[str, str]] = []

# Debug info from the last generation call — read by the UI console
_last_debug_info: Dict[str, Any] = {}


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


def get_current_chat_model() -> Tuple[Optional[str], bool]:
    """Get the currently loaded chat model key and loaded status."""
    return _current_chat_model_key, _current_chat_model_id is not None


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


def clear_conversation():
    """Clear the conversation history."""
    global _conversation_history
    _conversation_history = []


def get_conversation_history() -> List[Dict[str, str]]:
    """Get the current conversation history."""
    return _conversation_history.copy()


def parse_message_with_actions(message: str) -> Dict:
    """
    Parse a message for *action* markers and dialog.

    Returns:
        {
            "has_actions": bool,
            "has_dialog": bool,
            "action_parts": List[str],  # extracted *action* text (without asterisks)
            "dialog_parts": List[str],  # non-action text
            "segments": List[Dict],     # ordered list of {"type": "action"|"dialog", "text": str}
            "formatted": str,           # markdown formatted version
            "plain": str,               # plain text with actions preserved as *...*
            "display_text": str         # what to show in UI (same as plain)
        }

    Examples:
        "*looks around confused*"
        -> action_parts: ["looks around confused"]

        "I don't understand. *nervous laugh*"
        -> dialog_parts: ["I don't understand."]
        -> action_parts: ["nervous laugh"]

        "*picks up the sword and examines it* That's a fine weapon."
        -> action_parts: ["picks up the sword and examines it"]
        -> dialog_parts: ["That's a fine weapon."]
    """
    if not message:
        return {
            "has_actions": False,
            "has_dialog": False,
            "action_parts": [],
            "dialog_parts": [],
            "segments": [],
            "formatted": "",
            "plain": "",
            "display_text": "",
        }

    segments = []
    action_parts = []
    dialog_parts = []

    # Split on *...* patterns, keeping delimiters
    parts = re.split(r'(\*[^*]+\*)', message)
    for part in parts:
        if not part:
            continue
        action_match = re.fullmatch(r'\*([^*]+)\*', part)
        if action_match:
            action_text = action_match.group(1).strip()
            if action_text:
                segments.append({"type": "action", "text": action_text})
                action_parts.append(action_text)
        else:
            dialog_text = part.strip()
            if dialog_text:
                segments.append({"type": "dialog", "text": dialog_text})
                dialog_parts.append(dialog_text)

    has_actions = len(action_parts) > 0
    has_dialog = len(dialog_parts) > 0

    # Use the original message as plain text (actions are already in *...* format)
    plain = message

    # Formatted uses same format (markdown *italic*)
    formatted = plain

    return {
        "has_actions": has_actions,
        "has_dialog": has_dialog,
        "action_parts": action_parts,
        "dialog_parts": dialog_parts,
        "segments": segments,
        "formatted": formatted,
        "plain": plain,
        "display_text": plain,
    }


def build_message_with_directions(dialog: str, directions: str) -> Tuple[str, Optional[str]]:
    """
    Build the final message and optional system prompt addendum to send to the model,
    incorporating directions as a system-level instruction without making them part of dialog.

    Args:
        dialog: What the user/character is saying (supports *actions*)
        directions: Optional meta-instructions for how to respond

    Returns:
        Tuple of (user_message, directions_block or None)
        - user_message: The dialog text to send as the user turn
        - directions_block: Text to append to the system prompt for this turn, or None
    """
    parsed = parse_message_with_actions(dialog)
    user_message = parsed["plain"] if parsed["plain"] else dialog

    directions_block = None
    if directions and directions.strip():
        directions_block = (
            f"\n[CURRENT SCENE INSTRUCTIONS]\n{directions.strip()}\n[/CURRENT SCENE INSTRUCTIONS]"
        )

    return user_message, directions_block


def _render_messages_for_debug(messages: List[Dict[str, str]]) -> str:
    """Render a messages list as readable text for the debug console."""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


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
    directions: str = "",
    extra_stop_phrases: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Generate a chat response via Ollama.

    Args:
        user_message: The user's message (supports *action* markers)
        system_prompt: Optional system prompt for AI behavior
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        progress_callback: Unused (Ollama call here is non-streaming); kept for signature compatibility
        use_history: Whether to use/update conversation history (default True)
        directions: Optional meta-instructions for character behavior (injected into
                    system prompt for this turn only; not stored in history)
        extra_stop_phrases: Optional list of additional stop phrases to truncate
                           response (e.g., character/persona names to prevent the
                           model from writing dialog for both sides)

    Returns:
        Tuple of (success: bool, response: str)
    """
    global _conversation_history

    if _current_chat_model_id is None:
        return False, "No chat model loaded. Please load a model first."

    try:
        # Parse actions and build message/directions block
        parsed_message, directions_block = build_message_with_directions(user_message, directions)

        # Inject directions into system prompt for this turn only
        effective_system = system_prompt
        if directions_block:
            effective_system = (system_prompt or "") + directions_block

        history_to_use = _conversation_history if use_history else []

        messages: List[Dict[str, str]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.extend(history_to_use)
        messages.append({"role": "user", "content": parsed_message})

        stop_phrases = list(extra_stop_phrases) if extra_stop_phrases else []

        # Capture debug info
        _last_debug_info.clear()
        _last_debug_info["system_prompt"] = effective_system or ""
        _last_debug_info["user_message"] = parsed_message
        _last_debug_info["history_length"] = len(history_to_use)
        _last_debug_info["history_turns"] = list(history_to_use)
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

        if use_history:
            _conversation_history.append({"role": "user", "content": parsed_message})
            _conversation_history.append({"role": "assistant", "content": response})

            # Keep history manageable (last 10 exchanges)
            if len(_conversation_history) > 20:
                _conversation_history = _conversation_history[-20:]

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
    if _current_chat_model_id is None:
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

    stop_phrases = ["\nUser:", "\nAssistant:", "\n\n\n"]

    try:
        result = ollama_client.chat(
            _current_chat_model_id,
            [{"role": "user", "content": extraction_prompt}],
            options={
                "temperature": 0.2,
                "top_p": 0.85,
                "num_predict": 150,
                "stop": stop_phrases,
            },
        )
        response = (result.get("message") or {}).get("content", "").strip()

        if "NONE" in response.upper() and len(response) < 20:
            return []

        # Parse bullet points
        memories = []
        for line in response.split("\n"):
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
