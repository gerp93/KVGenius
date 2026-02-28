"""
Unit tests for src/image/enhancer.py
"""
import pytest

# conftest.py already mocked torch before this import
from src.image.enhancer import (
    count_clip_tokens,
    format_token_count,
    CLIP_MAX_TOKENS,
    PromptEnhancer,
)


class TestCountClipTokens:
    def test_empty_string_returns_zero(self):
        assert count_clip_tokens("") == 0

    def test_none_returns_zero(self):
        assert count_clip_tokens(None) == 0

    def test_single_word(self):
        result = count_clip_tokens("cat")
        assert result > 0

    def test_longer_text_more_tokens(self):
        short = count_clip_tokens("cat")
        longer = count_clip_tokens("a beautiful orange cat sitting on a wooden fence")
        assert longer > short

    def test_with_tokenizer_fallback(self):
        """If the tokenizer raises, should fall back gracefully."""
        class _BadTokenizer:
            def __call__(self, *a, **kw):
                raise RuntimeError("oops")

        result = count_clip_tokens("hello world", tokenizer=_BadTokenizer())
        assert result > 0  # Should fall back to estimation

    def test_with_working_tokenizer(self):
        """If a real tokenizer-like object is passed, use its count."""
        import types
        tok_result = types.SimpleNamespace(input_ids=types.SimpleNamespace(shape=(1, 10)))

        class _GoodTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None):
                return tok_result

        result = count_clip_tokens("any text", tokenizer=_GoodTokenizer())
        assert result == 10


class TestFormatTokenCount:
    def test_empty_prompt_shows_zero(self):
        output = format_token_count("")
        assert "0" in output

    def test_whitespace_only_shows_zero(self):
        output = format_token_count("   ")
        assert "0" in output

    def test_short_prompt_normal_format(self):
        output = format_token_count("a cat")
        assert "77" in output

    def test_over_limit_shows_warning(self, monkeypatch):
        # Monkeypatch count_clip_tokens to return a value above CLIP_MAX_TOKENS
        import src.image.enhancer as enhancer_mod
        monkeypatch.setattr(enhancer_mod, "count_clip_tokens", lambda text, tokenizer=None: 100)
        output = format_token_count("anything")
        assert "TRUNCATED" in output or "⚠" in output

    def test_near_limit_shows_warning(self, monkeypatch):
        import src.image.enhancer as enhancer_mod
        monkeypatch.setattr(enhancer_mod, "count_clip_tokens", lambda text, tokenizer=None: 65)
        output = format_token_count("anything")
        assert "close" in output or "⚠" in output


class TestPromptEnhancerInit:
    def test_default_init(self):
        pe = PromptEnhancer()
        assert pe.model is None
        assert pe.tokenizer is None

    def test_custom_cache_dir(self):
        pe = PromptEnhancer(cache_dir="/tmp/my_cache")
        assert pe.cache_dir == "/tmp/my_cache"


class TestDetectStyle:
    def test_anime_detection(self):
        pe = PromptEnhancer()
        assert pe.detect_style("anime girl with sword") == "anime"

    def test_fantasy_detection(self):
        pe = PromptEnhancer()
        assert pe.detect_style("a wizard casting a spell") == "fantasy"

    def test_portrait_detection(self):
        pe = PromptEnhancer()
        assert pe.detect_style("close up portrait of a woman") == "portrait"

    def test_artistic_detection(self):
        pe = PromptEnhancer()
        assert pe.detect_style("a beautiful painting of a landscape") == "artistic"

    def test_default_photorealistic(self):
        pe = PromptEnhancer()
        assert pe.detect_style("a dog in the park") == "photorealistic"


class TestEnhanceWithTemplate:
    def test_returns_tuple(self):
        pe = PromptEnhancer()
        result = pe.enhance_with_template("a cat")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_enhanced_prompt_contains_original(self):
        pe = PromptEnhancer()
        enhanced, _ = pe.enhance_with_template("a cat", style="photorealistic")
        assert "a cat" in enhanced

    def test_style_suffix_added(self):
        pe = PromptEnhancer()
        enhanced, _ = pe.enhance_with_template("a cat", style="anime")
        assert "anime" in enhanced.lower() or "cel shading" in enhanced.lower()

    def test_portrait_prefix_added(self):
        pe = PromptEnhancer()
        enhanced, _ = pe.enhance_with_template("young woman", style="portrait")
        assert enhanced.lower().startswith("portrait of")

    def test_portrait_prefix_not_doubled(self):
        pe = PromptEnhancer()
        enhanced, _ = pe.enhance_with_template("portrait of a man", style="portrait")
        assert enhanced.lower().count("portrait of") == 1

    def test_empty_prompt_returns_error(self):
        pe = PromptEnhancer()
        _, status = pe.enhance_with_template("")
        assert "❌" in status

    def test_whitespace_prompt_returns_error(self):
        pe = PromptEnhancer()
        _, status = pe.enhance_with_template("   ")
        assert "❌" in status

    def test_unknown_style_falls_back_to_photorealistic(self):
        pe = PromptEnhancer()
        enhanced, status = pe.enhance_with_template("a scene", style="nonexistent_style")
        assert enhanced != ""


class TestEnhanceSmart:
    def test_person_prompt_gets_face_details(self):
        pe = PromptEnhancer()
        enhanced, _ = pe.enhance_smart("a woman in the rain", style="photorealistic")
        assert "face" in enhanced.lower() or "eyes" in enhanced.lower()

    def test_landscape_prompt_gets_lighting(self):
        pe = PromptEnhancer()
        enhanced, _ = pe.enhance_smart("mountain landscape at night", style="photorealistic")
        assert len(enhanced) > len("mountain landscape at night")

    def test_empty_prompt_returns_error(self):
        pe = PromptEnhancer()
        _, status = pe.enhance_smart("")
        assert "❌" in status

    def test_returns_tuple(self):
        pe = PromptEnhancer()
        result = pe.enhance_smart("a dog")
        assert isinstance(result, tuple) and len(result) == 2


class TestQuickEnhance:
    def test_auto_detects_and_enhances(self):
        pe = PromptEnhancer()
        enhanced, status = pe.quick_enhance("an anime character")
        assert isinstance(enhanced, str)
        assert len(enhanced) > 0

    def test_returns_tuple(self):
        pe = PromptEnhancer()
        result = pe.quick_enhance("a landscape")
        assert isinstance(result, tuple) and len(result) == 2


class TestEnhanceWithAI:
    def test_falls_back_to_template_when_model_load_fails(self):
        """When the AI model can't be loaded, should fall back to template-based enhancement."""
        pe = PromptEnhancer(cache_dir="/nonexistent/path")
        enhanced, status = pe.enhance_with_ai("a cat", style="photorealistic")
        assert isinstance(enhanced, str)
        assert len(enhanced) > 0

    def test_empty_prompt_returns_error(self):
        pe = PromptEnhancer()
        _, status = pe.enhance_with_ai("")
        assert "❌" in status


class TestAiEnhanceAlias:
    def test_ai_enhance_delegates(self):
        pe = PromptEnhancer(cache_dir="/nonexistent/path")
        enhanced, status = pe.ai_enhance("a forest")
        assert isinstance(enhanced, str)
