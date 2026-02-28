"""
Unit tests for helper functions in src/image/generator.py
(read_image_metadata, format_image_metadata)
"""
import sys
import types
import pytest

# conftest.py has already stubbed torch and PIL; now refine the PIL.Image stub
# so read_image_metadata can simulate images with and without metadata.
import PIL.Image as _pil_image_mod

_real_open = getattr(_pil_image_mod, "_real_open", None)


class _FakeImageWithMeta:
    def __init__(self, info):
        self.info = info


class _FakeImageNoMeta:
    # No .info attribute at all
    pass


# Import the functions under test directly from the module (avoid __init__ re-export)
from src.image.generator import read_image_metadata, format_image_metadata


class TestReadImageMetadata:
    def test_returns_dict(self, monkeypatch, tmp_path):
        fake = _FakeImageWithMeta({
            "prompt": "a cat",
            "steps": "25",
            "model": "dreamshaper-8",
        })
        monkeypatch.setattr(_pil_image_mod, "open", lambda p: fake)

        result = read_image_metadata(str(tmp_path / "test.png"))
        assert isinstance(result, dict)
        assert result["prompt"] == "a cat"
        assert result["steps"] == "25"

    def test_missing_keys_not_included(self, monkeypatch, tmp_path):
        fake = _FakeImageWithMeta({"prompt": "only prompt here"})
        monkeypatch.setattr(_pil_image_mod, "open", lambda p: fake)

        result = read_image_metadata(str(tmp_path / "x.png"))
        assert "prompt" in result
        assert "model" not in result

    def test_no_info_attribute_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_pil_image_mod, "open", lambda p: _FakeImageNoMeta())
        result = read_image_metadata(str(tmp_path / "x.png"))
        assert result == {}

    def test_exception_returns_empty_dict(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_pil_image_mod, "open", lambda p: (_ for _ in ()).throw(IOError("no file")))
        result = read_image_metadata("/nonexistent/path.png")
        assert result == {}


class TestFormatImageMetadata:
    def test_empty_metadata_returns_no_metadata_message(self):
        result = format_image_metadata({})
        assert "No metadata" in result

    def test_prompt_included(self):
        result = format_image_metadata({"prompt": "a dog"})
        assert "a dog" in result

    def test_negative_prompt_included(self):
        result = format_image_metadata({"negative_prompt": "blurry"})
        assert "blurry" in result

    def test_model_included(self):
        result = format_image_metadata({"model": "dreamshaper-8"})
        assert "dreamshaper-8" in result

    def test_settings_line_shows_steps_and_cfg(self):
        result = format_image_metadata({
            "steps": "25",
            "guidance_scale": "7.5",
            "width": "512",
            "height": "768",
            "seed": "42",
        })
        assert "25" in result
        assert "7.5" in result
        assert "512" in result
        assert "42" in result

    def test_lora_shown_with_strength(self):
        result = format_image_metadata({
            "lora": "my_lora",
            "lora_strength": "0.8",
        })
        assert "my_lora" in result
        assert "0.8" in result

    def test_lora_none_not_shown(self):
        result = format_image_metadata({"lora": "None"})
        assert "None" not in result or "LoRA" not in result

    def test_generated_at_shown(self):
        result = format_image_metadata({"generated_at": "2025-01-01"})
        assert "2025-01-01" in result

    def test_full_metadata(self):
        meta = {
            "prompt": "a cat",
            "negative_prompt": "blurry",
            "model": "test-model",
            "steps": "30",
            "guidance_scale": "8.0",
            "width": "512",
            "height": "768",
            "seed": "1234",
            "lora": "test_lora",
            "lora_strength": "0.7",
            "generated_at": "2025-06-01",
        }
        result = format_image_metadata(meta)
        assert "a cat" in result
        assert "test-model" in result
        assert "test_lora" in result
