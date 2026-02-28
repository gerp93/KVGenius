"""
Unit tests for src/image/presets.py
"""
import os
import tempfile
import pytest
import yaml

# Import directly to avoid the __init__.py that pulls in ImageGenerator (torch dep)
from src.image.presets import (
    ENHANCEMENT_TEMPLATES,
    DEFAULT_IMAGE_MODELS,
    load_image_model_presets,
    build_available_models,
)


class TestEnhancementTemplates:
    def test_expected_styles_present(self):
        for style in ("photorealistic", "artistic", "portrait", "fantasy", "anime"):
            assert style in ENHANCEMENT_TEMPLATES

    def test_each_style_has_required_keys(self):
        for style, tpl in ENHANCEMENT_TEMPLATES.items():
            assert "prefix" in tpl, f"Missing 'prefix' in {style}"
            assert "suffix" in tpl, f"Missing 'suffix' in {style}"
            assert "quality" in tpl, f"Missing 'quality' in {style}"

    def test_portrait_prefix(self):
        assert ENHANCEMENT_TEMPLATES["portrait"]["prefix"] == "portrait of "


class TestDefaultImageModels:
    def test_default_models_exist(self):
        assert len(DEFAULT_IMAGE_MODELS) > 0

    def test_each_model_has_required_keys(self):
        required = {"id", "description", "vram", "default_steps", "default_guidance",
                    "default_width", "default_height", "step_range", "guidance_range",
                    "default_negative", "tips"}
        for name, model in DEFAULT_IMAGE_MODELS.items():
            missing = required - set(model.keys())
            assert not missing, f"Model '{name}' missing keys: {missing}"


class TestLoadImageModelPresets:
    def test_missing_file_returns_empty_dicts(self, tmp_path):
        models, templates = load_image_model_presets(config_dir=str(tmp_path))
        assert models == {}
        assert templates == {}

    def test_loads_from_yaml(self, tmp_path):
        preset_data = {
            "models": {
                "TestModel": {
                    "id": "org/test-model",
                    "description": "A test model",
                    "vram": "~4 GB",
                    "default_steps": 20,
                    "default_guidance": 7.0,
                    "default_width": 512,
                    "default_height": 512,
                    "step_range": [10, 50],
                    "guidance_range": [1.0, 15.0],
                    "default_negative": "blurry",
                    "tips": "use it wisely",
                }
            },
            "prompt_templates": {"test_template": {"prefix": "test "}},
        }
        yaml_path = tmp_path / "image_model_presets.yaml"
        yaml_path.write_text(yaml.dump(preset_data))

        models, templates = load_image_model_presets(config_dir=str(tmp_path))
        assert "TestModel" in models
        assert "test_template" in templates

    def test_malformed_yaml_returns_empty_dicts(self, tmp_path):
        bad_yaml = tmp_path / "image_model_presets.yaml"
        bad_yaml.write_text(":::bad yaml:::")
        models, templates = load_image_model_presets(config_dir=str(tmp_path))
        assert models == {}
        assert templates == {}


class TestBuildAvailableModels:
    def test_empty_presets_returns_defaults(self):
        models = build_available_models(presets={})
        assert models == DEFAULT_IMAGE_MODELS

    def test_none_presets_loads_from_config(self):
        # Should not raise; may return defaults or actual config values
        models = build_available_models(presets=None)
        assert isinstance(models, dict)

    def test_custom_presets_returned(self):
        custom = {
            "MyModel": {
                "id": "org/my-model",
                "description": "Custom",
                "vram": "~6 GB",
                "default_steps": 30,
                "default_guidance": 8.0,
                "default_width": 768,
                "default_height": 768,
                "step_range": [1, 50],
                "guidance_range": [0.0, 20.0],
                "default_negative": "ugly",
                "tips": "n/a",
            }
        }
        models = build_available_models(presets=custom)
        assert "MyModel" in models
        assert models["MyModel"]["id"] == "org/my-model"

    def test_missing_fields_get_defaults(self):
        minimal = {"MinModel": {"id": "x/y"}}
        models = build_available_models(presets=minimal)
        m = models["MinModel"]
        assert m["default_steps"] == 25
        assert m["default_guidance"] == 7.5
