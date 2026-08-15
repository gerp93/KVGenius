"""
Unit tests for src/chat/chatbot.py
"""
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Helpers – build a lightweight mock model and tokenizer
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal tokenizer stub."""
    eos_token_id = 2
    pad_token_id = 1
    chat_template = None  # triggers the fallback path in build_prompt

    def encode(self, text, return_tensors=None, truncation=False, max_length=None):
        # Return a simple fake tensor
        import sys
        torch = sys.modules["torch"]
        t = torch.Tensor([1, 2, 3])
        t.shape = (1, 3)
        return t

    def decode(self, token_ids, skip_special_tokens=False):
        return "Hello, I am the assistant."

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise NotImplementedError("not available")


class _MockChatTemplateTokenizer(_MockTokenizer):
    """Tokenizer that has a working chat_template."""
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        return "".join(parts)


class _MockModel:
    """Minimal model stub whose generate() returns a fake tensor."""

    def generate(self, inputs, **kwargs):
        # Return something that looks like a 2-D tensor
        import sys
        torch = sys.modules["torch"]
        t = torch.Tensor([1, 2, 3, 4, 5])
        t.shape = (1, 5)

        def getitem(idx):
            inner = torch.Tensor([4, 5])
            inner.shape = (2,)
            return inner

        t.__getitem__ = getitem
        return t


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from src.chat.chatbot import ChatBot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatBotInit:
    def test_default_attributes(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        assert bot.max_history == 5
        assert bot.device == "cpu"
        assert bot.conversation_history == []

    def test_custom_max_history(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer(), max_history=3)
        assert bot.max_history == 3


class TestAddToHistory:
    def test_add_single_message(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        bot.add_to_history("user", "Hello")
        assert len(bot.conversation_history) == 1
        assert bot.conversation_history[0] == {"role": "user", "content": "Hello"}

    def test_history_trimmed_to_max(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer(), max_history=2)
        # Add more than max_history * 2 messages (5 pairs = 10 messages)
        for i in range(6):
            bot.add_to_history("user", f"msg{i}")
            bot.add_to_history("assistant", f"reply{i}")
        # Should be trimmed to max_history * 2 = 4
        assert len(bot.conversation_history) <= 4

    def test_get_history_returns_copy(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        bot.add_to_history("user", "Hi")
        history = bot.get_history()
        assert len(history) == 1


class TestBuildPrompt:
    def test_fallback_format_no_history(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        prompt = bot.build_prompt("What is 2+2?")
        assert "User: What is 2+2?" in prompt
        assert "Assistant:" in prompt

    def test_fallback_format_with_system_prompt(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        prompt = bot.build_prompt("Hi", system_prompt="You are helpful.")
        assert "You are helpful." in prompt
        assert "User: Hi" in prompt

    def test_fallback_includes_conversation_history(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        bot.add_to_history("user", "First question")
        bot.add_to_history("assistant", "First answer")
        prompt = bot.build_prompt("Second question")
        assert "First question" in prompt
        assert "First answer" in prompt
        assert "Second question" in prompt

    def test_chat_template_used_when_available(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockChatTemplateTokenizer())
        prompt = bot.build_prompt("Hello")
        assert "<user>Hello</user>" in prompt


class TestGenerateResponse:
    def test_returns_string(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        response = bot.generate_response("Hello")
        assert isinstance(response, str)

    def test_history_updated_after_response(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        bot.generate_response("Hello")
        history = bot.get_history()
        # Should have user and assistant turns
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    def test_stop_phrases_cleaned(self):
        """Ensure stop-phrase cleaning logic is exercised (mocked response)."""
        class _StopPhraseTokenizer(_MockTokenizer):
            def decode(self, token_ids, skip_special_tokens=False):
                return "Hello\nUser: someone"

        bot = ChatBot(model=_MockModel(), tokenizer=_StopPhraseTokenizer())
        response = bot.generate_response("Hi")
        assert "\nUser:" not in response


class TestResetConversation:
    def test_clears_history(self):
        bot = ChatBot(model=_MockModel(), tokenizer=_MockTokenizer())
        bot.add_to_history("user", "Hello")
        bot.reset_conversation()
        assert bot.conversation_history == []
