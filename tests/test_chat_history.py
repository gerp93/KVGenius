"""
Unit tests for src/database/chat_history.py
"""
import os
import tempfile
import pytest

from src.database.chat_history import ChatHistoryDB


@pytest.fixture()
def db(tmp_path):
    """Return a ChatHistoryDB backed by a temporary file."""
    db_file = str(tmp_path / "test_chat.db")
    return ChatHistoryDB(db_path=db_file)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

class TestConversations:
    def test_create_conversation_returns_int(self, db):
        conv_id = db.create_conversation(model="test-model")
        assert isinstance(conv_id, int)
        assert conv_id > 0

    def test_get_conversation(self, db):
        conv_id = db.create_conversation(model="gpt-4", title="My Chat")
        conv = db.get_conversation(conv_id)
        assert conv is not None
        assert conv["model"] == "gpt-4"
        assert conv["title"] == "My Chat"

    def test_get_conversation_missing(self, db):
        assert db.get_conversation(9999) is None

    def test_update_conversation_title(self, db):
        conv_id = db.create_conversation(model="m")
        db.update_conversation_title(conv_id, "New Title")
        conv = db.get_conversation(conv_id)
        assert conv["title"] == "New Title"

    def test_get_recent_conversations(self, db):
        for i in range(5):
            db.create_conversation(model=f"model-{i}", title=f"Chat {i}")
        recent = db.get_recent_conversations(limit=3)
        assert len(recent) == 3

    def test_delete_conversation(self, db):
        conv_id = db.create_conversation(model="m")
        db.delete_conversation(conv_id)
        assert db.get_conversation(conv_id) is None

    def test_delete_all_conversations(self, db):
        db.create_conversation(model="a")
        db.create_conversation(model="b")
        db.delete_all_conversations()
        assert db.get_recent_conversations() == []


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class TestMessages:
    def test_add_and_retrieve_messages(self, db):
        conv_id = db.create_conversation(model="m")
        db.add_message(conv_id, "user", "Hello!")
        db.add_message(conv_id, "assistant", "Hi there!")
        history = db.get_conversation_messages(conv_id)
        assert len(history) == 1  # one (user, assistant) pair
        assert history[0] == ("Hello!", "Hi there!")

    def test_delete_all_conversations_removes_messages(self, db):
        """delete_all_conversations explicitly removes messages too."""
        conv_id = db.create_conversation(model="m")
        db.add_message(conv_id, "user", "msg")
        db.add_message(conv_id, "assistant", "reply")
        db.delete_all_conversations()
        assert db.get_conversation_messages(conv_id) == []

    def test_unmatched_user_message_not_returned(self, db):
        """A user message without a following assistant reply is not in the history."""
        conv_id = db.create_conversation(model="m")
        db.add_message(conv_id, "user", "orphan")
        history = db.get_conversation_messages(conv_id)
        assert history == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExportConversation:
    def test_export_contains_messages(self, db):
        conv_id = db.create_conversation(model="m", title="Export Test")
        db.add_message(conv_id, "user", "Q?")
        db.add_message(conv_id, "assistant", "A.")
        export = db.export_conversation(conv_id)
        assert export["title"] == "Export Test"
        assert len(export["messages"]) == 2


# ---------------------------------------------------------------------------
# AI Characters
# ---------------------------------------------------------------------------

class TestAICharacters:
    def test_create_and_get_character(self, db):
        char_id = db.create_ai_character(name="TestBot", system_prompt="Be helpful")
        char = db.get_ai_character(char_id)
        assert char is not None
        assert char["name"] == "TestBot"
        assert char["system_prompt"] == "Be helpful"

    def test_get_all_ai_characters(self, db):
        db.create_ai_character(name="Alpha")
        db.create_ai_character(name="Beta")
        chars = db.get_all_ai_characters()
        names = [c["name"] for c in chars]
        assert "Alpha" in names
        assert "Beta" in names

    def test_update_ai_character(self, db):
        char_id = db.create_ai_character(name="Original")
        db.update_ai_character(char_id, name="Updated", temperature=0.5)
        char = db.get_ai_character(char_id)
        assert char["name"] == "Updated"
        assert char["temperature"] == 0.5

    def test_delete_ai_character(self, db):
        char_id = db.create_ai_character(name="ToDelete")
        db.delete_ai_character(char_id)
        assert db.get_ai_character(char_id) is None

    def test_get_missing_character_returns_none(self, db):
        assert db.get_ai_character(9999) is None


# ---------------------------------------------------------------------------
# User Personas
# ---------------------------------------------------------------------------

class TestUserPersonas:
    def test_create_and_get_persona(self, db):
        persona_id = db.create_user_persona(name="Hero", description="A brave hero")
        persona = db.get_user_persona(persona_id)
        assert persona is not None
        assert persona["name"] == "Hero"
        assert persona["description"] == "A brave hero"

    def test_get_all_user_personas(self, db):
        db.create_user_persona(name="PersonaA")
        db.create_user_persona(name="PersonaB")
        personas = db.get_all_user_personas()
        names = [p["name"] for p in personas]
        assert "PersonaA" in names
        assert "PersonaB" in names

    def test_get_missing_persona_returns_none(self, db):
        assert db.get_user_persona(9999) is None


# ---------------------------------------------------------------------------
# Prompt Library
# ---------------------------------------------------------------------------

class TestPromptLibrary:
    def test_save_and_get_prompt(self, db):
        prompt_id = db.save_prompt(
            name="Sunset",
            prompt="A beautiful sunset over mountains",
            negative_prompt="blurry",
        )
        assert isinstance(prompt_id, int)
        prompt = db.get_prompt_by_id(prompt_id)
        assert prompt["prompt"] == "A beautiful sunset over mountains"
        assert prompt["negative_prompt"] == "blurry"

    def test_save_prompt_upsert(self, db):
        """Saving same name twice should update, not create duplicate."""
        db.save_prompt(name="Same", prompt="original")
        db.save_prompt(name="Same", prompt="updated")
        all_prompts = db.get_all_prompts()
        same = [p for p in all_prompts if p["name"] == "Same"]
        assert len(same) == 1
        assert same[0]["prompt"] == "updated"

    def test_get_prompt_by_name(self, db):
        db.save_prompt(name="MyPrompt", prompt="test prompt")
        p = db.get_prompt_by_name("MyPrompt")
        assert p is not None
        assert p["prompt"] == "test prompt"

    def test_get_all_prompts(self, db):
        db.save_prompt(name="P1", prompt="first")
        db.save_prompt(name="P2", prompt="second")
        prompts = db.get_all_prompts()
        assert len(prompts) >= 2

    def test_delete_prompt(self, db):
        prompt_id = db.save_prompt(name="ToDelete", prompt="bye")
        assert db.delete_prompt(prompt_id) is True
        assert db.get_prompt_by_id(prompt_id) is None

    def test_delete_nonexistent_prompt(self, db):
        assert db.delete_prompt(9999) is False

    def test_update_prompt(self, db):
        prompt_id = db.save_prompt(name="Upd", prompt="old prompt")
        db.update_prompt(prompt_id, prompt="new prompt")
        p = db.get_prompt_by_id(prompt_id)
        assert p["prompt"] == "new prompt"

    def test_categories_round_trip(self, db):
        db.save_prompt(name="CatTest", prompt="x", categories=["portrait", "nature"])
        # get_all_prompts deserializes categories; get_prompt_by_name returns raw JSON
        prompts = db.get_all_prompts()
        p = next(x for x in prompts if x["name"] == "CatTest")
        assert isinstance(p["categories"], list)
        assert "portrait" in p["categories"]


# ---------------------------------------------------------------------------
# init_defaults
# ---------------------------------------------------------------------------

class TestInitDefaults:
    def test_creates_default_characters_and_personas(self, db):
        db.init_defaults()
        chars = db.get_all_ai_characters()
        personas = db.get_all_user_personas()
        assert len(chars) > 0
        assert len(personas) > 0

    def test_init_defaults_idempotent(self, db):
        db.init_defaults()
        db.init_defaults()
        chars = db.get_all_ai_characters()
        # Should not have duplicates
        names = [c["name"] for c in chars]
        assert len(names) == len(set(names))
