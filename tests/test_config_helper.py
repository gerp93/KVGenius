"""
Unit tests for src/utils/config_helper.py
"""
import json
import os
import tempfile
import pytest

from src.utils.config_helper import (
    load_config,
    get_default_config,
    setup_logging,
    load_environment,
    save_chat_history,
    load_chat_history,
)


class TestGetDefaultConfig:
    def test_returns_dict(self):
        config = get_default_config()
        assert isinstance(config, dict)

    def test_has_model_section(self):
        config = get_default_config()
        assert "model" in config
        assert "name" in config["model"]

    def test_has_generation_section(self):
        config = get_default_config()
        assert "generation" in config
        assert "temperature" in config["generation"]

    def test_has_chat_section(self):
        config = get_default_config()
        assert "chat" in config
        assert "max_history" in config["chat"]

    def test_has_app_section(self):
        config = get_default_config()
        assert "app" in config


class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_config(str(tmp_path / "nonexistent.yaml"))
        assert config == get_default_config()

    def test_loads_yaml_file(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump({"my_key": "my_value"}))
        result = load_config(str(cfg_file))
        assert result["my_key"] == "my_value"

    def test_malformed_yaml_returns_defaults(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_bytes(b"\x00\x01\x02 not valid yaml \xff")
        result = load_config(str(bad_file))
        assert result == get_default_config()


class TestSetupLogging:
    def test_does_not_raise(self):
        setup_logging("INFO")
        setup_logging("DEBUG")
        setup_logging("WARNING")


class TestLoadEnvironment:
    def test_does_not_raise_without_env_file(self, tmp_path, monkeypatch):
        """load_environment should not raise when no .env exists."""
        monkeypatch.chdir(tmp_path)
        load_environment()  # Should complete silently

    def test_loads_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=hello\n")
        monkeypatch.chdir(tmp_path)
        load_environment()
        # After loading, TEST_VAR should be in environment
        assert os.environ.get("TEST_VAR") == "hello"


class TestSaveChatHistory:
    def test_saves_json_file(self, tmp_path):
        history = [{"role": "user", "content": "Hello"}]
        filepath = str(tmp_path / "history.json")
        save_chat_history(history, filepath)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == history

    def test_save_empty_history(self, tmp_path):
        filepath = str(tmp_path / "empty.json")
        save_chat_history([], filepath)
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == []

    def test_save_does_not_raise_on_bad_path(self):
        """Should log error but not raise."""
        save_chat_history([], "/nonexistent_dir/history.json")


class TestLoadChatHistory:
    def test_loads_saved_history(self, tmp_path):
        history = [{"role": "user", "content": "Test"}]
        filepath = str(tmp_path / "h.json")
        with open(filepath, "w") as f:
            json.dump(history, f)
        result = load_chat_history(filepath)
        assert result == history

    def test_returns_empty_list_if_missing(self, tmp_path):
        result = load_chat_history(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_returns_empty_list_on_bad_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{")
        result = load_chat_history(str(bad_file))
        assert result == []
