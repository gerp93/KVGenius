"""
Unit tests for core/config.py
"""
import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch


# We need to import carefully because core/config.py runs code at module level
# (creates directories, loads presets). We patch the filesystem operations.
import core.config as core_config


class TestLoadConfig:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        result = core_config.load_config("nonexistent.yaml")
        assert result == {}

    def test_loads_yaml_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"key": "value"}))
        result = core_config.load_config("test.yaml")
        assert result["key"] == "value"

    def test_falls_back_to_example_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        example_file = tmp_path / "settings.example.yaml"
        example_file.write_text(yaml.dump({"from_example": True}))
        result = core_config.load_config("settings.yaml")
        assert result.get("from_example") is True

    def test_empty_yaml_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        result = core_config.load_config("empty.yaml")
        assert result == {}


class TestSaveConfig:
    def test_saves_yaml_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        data = {"foo": "bar", "count": 42}
        result = core_config.save_config("output.yaml", data)
        assert result is True
        saved = yaml.safe_load((tmp_path / "output.yaml").read_text())
        assert saved["foo"] == "bar"
        assert saved["count"] == 42

    def test_returns_false_on_write_error(self, tmp_path, monkeypatch):
        # Point CONFIG_DIR to a file (not a directory) to trigger an error
        fake_dir = tmp_path / "not_a_dir.txt"
        fake_dir.write_text("i am a file")
        monkeypatch.setattr(core_config, "CONFIG_DIR", fake_dir)
        result = core_config.save_config("settings.yaml", {"x": 1})
        assert result is False


class TestLoadUserSettings:
    def test_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        result = core_config.load_user_settings()
        assert isinstance(result, dict)

    def test_returns_saved_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        (tmp_path / "settings.yaml").write_text(yaml.dump({"theme": "dark"}))
        result = core_config.load_user_settings()
        assert result.get("theme") == "dark"


class TestSaveUserSettings:
    def test_saves_and_reloads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        settings = {"language": "en", "debug": False}
        assert core_config.save_user_settings(settings) is True
        reloaded = core_config.load_user_settings()
        assert reloaded["language"] == "en"


class TestGetSetting:
    def test_returns_default_when_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        result = core_config.get_setting("nonexistent_key", default="fallback")
        assert result == "fallback"

    def test_returns_value_when_key_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        (tmp_path / "settings.yaml").write_text(yaml.dump({"my_key": "my_val"}))
        result = core_config.get_setting("my_key")
        assert result == "my_val"


class TestSetSetting:
    def test_sets_and_retrieves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        assert core_config.set_setting("color", "blue") is True
        assert core_config.get_setting("color") == "blue"

    def test_does_not_overwrite_other_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        core_config.set_setting("first", 1)
        core_config.set_setting("second", 2)
        assert core_config.get_setting("first") == 1
        assert core_config.get_setting("second") == 2


class TestLoadImageModelPresets:
    def test_returns_three_dicts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        hf, local, templates = core_config.load_image_model_presets()
        assert isinstance(hf, dict)
        assert isinstance(local, dict)
        assert isinstance(templates, dict)

    def test_loads_from_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        data = {
            "huggingface": {"Model A": {"id": "org/model-a"}},
            "local": {"Local B": {"path": "/local/b"}},
            "prompt_templates": {"cool": {"prefix": "cool "}},
        }
        (tmp_path / "image_model_presets.yaml").write_text(yaml.dump(data))
        hf, local, templates = core_config.load_image_model_presets()
        assert "Model A" in hf
        assert "Local B" in local
        assert "cool" in templates


class TestLoadChatModelPresets:
    def test_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        result = core_config.load_chat_model_presets()
        assert isinstance(result, dict)

    def test_loads_models_from_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_config, "CONFIG_DIR", tmp_path)
        data = {"models": {"TestChat": {"id": "org/test-chat", "vram": "~7 GB"}}}
        (tmp_path / "chat_model_presets.yaml").write_text(yaml.dump(data))
        result = core_config.load_chat_model_presets()
        assert "TestChat" in result
