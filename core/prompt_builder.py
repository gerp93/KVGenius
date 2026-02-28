"""
Prompt template builder for KVGenius chat.

Loads templates from config/prompt_templates.yaml, supports per-character
overrides stored in the database, and provides a single entry point for
assembling the full system prompt sent to the model.
"""
import logging
from typing import Dict, Any, Optional, List

from .config import load_config, save_config

logger = logging.getLogger(__name__)

CONFIG_FILE = "prompt_templates.yaml"

# Keys that map to configurable templates / values
TEMPLATE_KEYS = [
    "character_instructions_template",
    "persona_context_template",
    "directions_template",
    "memory_template",
]

SETTING_KEYS = [
    "stop_phrases",
    "use_character_name_as_stop",
    "use_persona_name_as_stop",
]

ALL_KEYS = TEMPLATE_KEYS + SETTING_KEYS

# Hard-coded defaults used when the YAML file is missing or a key is absent.
_BUILTIN_DEFAULTS: Dict[str, Any] = {
    "character_instructions_template": (
        '[CHARACTER RULES]\n'
        'You must ONLY write your own character\'s dialog and actions.\n'
        'NEVER write dialog, actions, or thoughts for the user or any other character.\n'
        'Wait for the user to provide their own words and actions.\n'
        '[/CHARACTER RULES]'
    ),
    "persona_context_template": (
        '[USER PERSONA]\n'
        'The user is playing a character named "{persona_name}". Address them as {persona_name}.\n'
        '{persona_name}\'s background: {persona_background}\n'
        '[/USER PERSONA]'
    ),
    "directions_template": (
        '[CURRENT SCENE INSTRUCTIONS]\n'
        '{directions}\n'
        '[/CURRENT SCENE INSTRUCTIONS]'
    ),
    "memory_template": (
        '[Key memories from this conversation - use these to maintain continuity:]\n'
        '{memories}'
    ),
    "stop_phrases": [
        "\nUser:",
        "\nAssistant:",
        "\n\n\n",
        "<|im_end|>",
        "</s>",
        "[INST]",
    ],
    "use_character_name_as_stop": True,
    "use_persona_name_as_stop": True,
}


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def load_prompt_templates() -> Dict[str, Any]:
    """Load prompt templates from config, falling back to built-in defaults."""
    cfg = load_config(CONFIG_FILE)
    result: Dict[str, Any] = {}
    for key in ALL_KEYS:
        val = cfg.get(key)
        if val is None:
            val = _BUILTIN_DEFAULTS.get(key)
        # YAML multi-line strings may have a trailing newline; strip it for templates
        if isinstance(val, str):
            val = val.strip()
        result[key] = val
    return result


def save_prompt_templates(templates: Dict[str, Any]) -> bool:
    """Save prompt templates back to config/prompt_templates.yaml."""
    # Load existing to preserve comments structure, then overwrite values
    cfg = load_config(CONFIG_FILE) or {}
    for key in ALL_KEYS:
        if key in templates:
            cfg[key] = templates[key]
    return save_config(CONFIG_FILE, cfg)


def get_template(key: str) -> Any:
    """Get a single template value by key."""
    templates = load_prompt_templates()
    return templates.get(key, _BUILTIN_DEFAULTS.get(key))


# ---------------------------------------------------------------------------
# Character override helpers
# ---------------------------------------------------------------------------

def apply_character_override(
    base_value: str,
    override_value: Optional[str],
    mode: str = "replace",
) -> str:
    """
    Apply a character-level override to a base template string.

    Args:
        base_value: The global default template text.
        override_value: The character's override text (may be None or empty).
        mode: "replace" — override replaces the base entirely.
              "append"  — override is appended after the base.

    Returns:
        The effective template string.
    """
    if not override_value or not override_value.strip():
        return base_value
    override_value = override_value.strip()
    if mode == "append":
        return f"{base_value}\n{override_value}"
    # default: replace
    return override_value


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles the full system prompt for a chat turn.

    Usage::

        builder = PromptBuilder()
        # Optionally load character overrides from DB
        builder.load_character_overrides(character_dict)
        system = builder.build_system_prompt(
            base_system_prompt="You are a helpful assistant.",
            persona_name="Explorer",
            persona_background="A brave explorer of unknown lands.",
            directions="Be mysterious and foreboding.",
            memories=["Found a hidden cave", "Met a stranger"],
        )
        stops = builder.get_stop_phrases(
            character_name="Wizard",
            persona_name="Explorer",
        )
    """

    def __init__(self):
        self._globals = load_prompt_templates()
        # Per-character overrides (set via load_character_overrides)
        self._char_overrides: Dict[str, Any] = {}

    # --- Public helpers to reload after UI edits --------------------------

    def reload_globals(self):
        """Re-read global templates from disk."""
        self._globals = load_prompt_templates()

    # --- Character overrides ----------------------------------------------

    def load_character_overrides(self, character: Optional[Dict[str, Any]]):
        """
        Load prompt-template overrides from a character dict (as returned by
        ``db.get_ai_character()``).

        The character dict may contain a JSON-decoded ``prompt_overrides`` key::

            {
                "persona_context_template": "...",
                "directions_template": "...",
                "memory_template": "...",
                "override_mode": "replace" | "amend",
            }
        """
        self._char_overrides = {}
        if not character:
            return
        overrides = character.get("prompt_overrides")
        if isinstance(overrides, dict):
            self._char_overrides = overrides
        elif isinstance(overrides, str):
            import json
            try:
                self._char_overrides = json.loads(overrides) if overrides.strip() else {}
            except (json.JSONDecodeError, ValueError):
                self._char_overrides = {}

    def _effective(self, key: str) -> str:
        """Return the effective template value for *key*, applying overrides."""
        base = self._globals.get(key, _BUILTIN_DEFAULTS.get(key, ""))
        if isinstance(base, str):
            base = base.strip()
        override = self._char_overrides.get(key)
        if override is not None:
            mode = self._char_overrides.get("override_mode", "replace")
            return apply_character_override(base, override, mode)
        return base

    # --- Building the prompt ----------------------------------------------

    def build_system_prompt(
        self,
        base_system_prompt: Optional[str] = None,
        persona_name: Optional[str] = None,
        persona_background: Optional[str] = None,
        directions: Optional[str] = None,
        memories: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Assemble the full system prompt for a single chat turn.

        Sections are only appended when their inputs are non-empty.

        Returns:
            The assembled system prompt string, or None if everything is empty.
        """
        parts: List[str] = []

        # 1. Base system prompt (character personality or manual input)
        if base_system_prompt:
            parts.append(base_system_prompt.strip())

        # 2. Character instructions (ALWAYS injected — baseline behavior rules)
        char_instructions = self._effective("character_instructions_template")
        if char_instructions and char_instructions.strip():
            parts.append(char_instructions.strip())

        # 3. Persona context (only when a persona is active)
        if persona_name and persona_background:
            tmpl = self._effective("persona_context_template")
            try:
                block = tmpl.format(
                    persona_name=persona_name,
                    persona_background=persona_background,
                )
                parts.append(block.strip())
            except KeyError as exc:
                logger.warning(f"persona_context_template placeholder error: {exc}")
                parts.append(tmpl)

        # 4. Memories
        if memories:
            tmpl = self._effective("memory_template")
            mem_lines = "\n".join(f"- {m}" for m in memories)
            try:
                block = tmpl.format(memories=mem_lines)
                parts.append(block.strip())
            except KeyError as exc:
                logger.warning(f"memory_template placeholder error: {exc}")
                parts.append(tmpl)

        # 5. Directions (transient, per-turn)
        if directions and directions.strip():
            tmpl = self._effective("directions_template")
            try:
                block = tmpl.format(directions=directions.strip())
                parts.append(block.strip())
            except KeyError as exc:
                logger.warning(f"directions_template placeholder error: {exc}")
                parts.append(tmpl)

        if not parts:
            return None
        return "\n\n".join(parts)

    # --- Stop phrases -----------------------------------------------------

    def get_stop_phrases(
        self,
        character_name: Optional[str] = None,
        persona_name: Optional[str] = None,
    ) -> List[str]:
        """
        Return the list of stop phrases for this turn, including any
        auto-generated ones from character/persona names.
        """
        base = list(self._globals.get("stop_phrases", _BUILTIN_DEFAULTS["stop_phrases"]))

        if self._globals.get("use_character_name_as_stop", True) and character_name:
            base.append(f"\n{character_name}:")

        if self._globals.get("use_persona_name_as_stop", True) and persona_name:
            if persona_name != "Myself (Default)":
                base.append(f"\n{persona_name}:")

        return base
