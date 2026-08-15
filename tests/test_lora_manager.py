"""
Unit tests for src/models/lora_manager.py
"""
import json
import os
import pytest

# Import directly (not via src.models package which would trigger model_loader/torch)
import importlib.util
import sys


def _import_lora_manager():
    spec = importlib.util.spec_from_file_location(
        "lora_manager_direct",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "models", "lora_manager.py"
        )
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lm = _import_lora_manager()

LoRAInfo = _lm.LoRAInfo
LoRAUseCase = _lm.LoRAUseCase
LoRABaseModel = _lm.LoRABaseModel
LoRAManager = _lm.LoRAManager


# ---------------------------------------------------------------------------
# LoRAInfo
# ---------------------------------------------------------------------------

class TestLoRAInfo:
    def test_to_dict(self):
        info = LoRAInfo(
            name="MyLoRA",
            path="/path/to/lora",
            use_case=LoRAUseCase.IMAGE,
            base_model=LoRABaseModel.SD15,
            trigger_words=["my_trigger"],
            default_weight=0.7,
        )
        d = info.to_dict()
        assert d["name"] == "MyLoRA"
        assert d["use_case"] == "image"
        assert d["base_model"] == "sd15"
        assert d["trigger_words"] == ["my_trigger"]
        assert d["default_weight"] == 0.7

    def test_from_dict(self):
        data = {
            "name": "FromDict",
            "use_case": "text",
            "base_model": "mistral",
            "trigger_words": ["tok1"],
            "compatible_models": [],
            "default_weight": 0.9,
            "category": "general",
            "description": "A LoRA",
            "preview_image": None,
            "rank": 16,
            "source": "civitai",
        }
        info = LoRAInfo.from_dict(data, path="/some/path")
        assert info.name == "FromDict"
        assert info.use_case == LoRAUseCase.TEXT
        assert info.base_model == LoRABaseModel.MISTRAL
        assert info.rank == 16

    def test_from_dict_trigger_word_compat(self):
        """from_dict should accept legacy 'trigger_word' key."""
        data = {
            "use_case": "image",
            "base_model": "sd15",
            "trigger_word": "single_trigger",
        }
        info = LoRAInfo.from_dict(data, path="/p")
        assert "single_trigger" in info.trigger_words

    def test_default_values(self):
        info = LoRAInfo(name="n", path="/p", use_case=LoRAUseCase.IMAGE, base_model=LoRABaseModel.UNKNOWN)
        assert info.default_weight == 0.8
        assert info.category == "general"
        assert info.trigger_words == []


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_use_case_values(self):
        assert LoRAUseCase.IMAGE.value == "image"
        assert LoRAUseCase.TEXT.value == "text"
        assert LoRAUseCase.CAH.value == "cah"

    def test_base_model_values(self):
        assert LoRABaseModel.SD15.value == "sd15"
        assert LoRABaseModel.SDXL.value == "sdxl"
        assert LoRABaseModel.MISTRAL.value == "mistral"
        assert LoRABaseModel.LLAMA.value == "llama"
        assert LoRABaseModel.LLAMA3.value == "llama3"
        assert LoRABaseModel.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# LoRAManager – scan_directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def manager(tmp_path):
    """Return a LoRAManager pointing at a temp directory (no config file)."""
    return LoRAManager(base_dir=str(tmp_path / "loras"), config_path=str(tmp_path / "none.yaml"))


class TestScanDirectory:
    def test_empty_directory_returns_empty_list(self, manager, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        loras = manager.scan_directory(str(empty_dir))
        assert loras == []

    def test_nonexistent_directory_returns_empty_list(self, manager):
        loras = manager.scan_directory("/nonexistent/path/xyz")
        assert loras == []

    def test_finds_lora_with_adapter_config(self, tmp_path, manager):
        lora_dir = tmp_path / "loras" / "my_lora"
        lora_dir.mkdir(parents=True)

        adapter_cfg = {
            "base_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "r": 16,
        }
        (lora_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg))

        loras = manager.scan_directory(str(tmp_path / "loras"))
        assert len(loras) == 1
        assert loras[0].name == "my_lora"
        assert loras[0].rank == 16

    def test_finds_lora_with_metadata(self, tmp_path, manager):
        lora_dir = tmp_path / "loras2" / "cool_lora"
        lora_dir.mkdir(parents=True)

        adapter_cfg = {"base_model_name_or_path": "stabilityai/sdxl-turbo"}
        (lora_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg))

        metadata = {
            "name": "Cool LoRA",
            "trigger_word": "cool",
            "description": "A cool LoRA",
            "category": "style",
            "source": "civitai",
        }
        (lora_dir / "lora_metadata.json").write_text(json.dumps(metadata))

        loras = manager.scan_directory(str(tmp_path / "loras2"))
        assert len(loras) == 1
        lora = loras[0]
        assert lora.name == "Cool LoRA"
        assert "cool" in lora.trigger_words
        assert lora.base_model == LoRABaseModel.SDXL

    def test_detects_preview_image(self, tmp_path, manager):
        lora_dir = tmp_path / "loras3" / "preview_lora"
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter_config.json").write_text(json.dumps({"r": 4}))
        (lora_dir / "preview.png").write_bytes(b"fake png")

        loras = manager.scan_directory(str(tmp_path / "loras3"))
        assert loras[0].preview_image is not None
        assert "preview.png" in loras[0].preview_image


# ---------------------------------------------------------------------------
# LoRAManager – detect helpers
# ---------------------------------------------------------------------------

class TestDetectHelpers:
    def test_detect_sd15_base_model(self, manager):
        cfg = {"base_model_name_or_path": "runwayml/stable-diffusion-v1-5"}
        result = manager._detect_base_model_from_adapter_config(cfg)
        assert result == LoRABaseModel.SD15

    def test_detect_sdxl_base_model(self, manager):
        cfg = {"base_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"}
        result = manager._detect_base_model_from_adapter_config(cfg)
        assert result == LoRABaseModel.SDXL

    def test_detect_mistral_base_model(self, manager):
        cfg = {"base_model_name_or_path": "mistralai/Mistral-7B-v0.1"}
        result = manager._detect_base_model_from_adapter_config(cfg)
        assert result == LoRABaseModel.MISTRAL

    def test_detect_llama3_base_model(self, manager):
        cfg = {"base_model_name_or_path": "meta-llama/Llama-3-8b"}
        result = manager._detect_base_model_from_adapter_config(cfg)
        assert result == LoRABaseModel.LLAMA3

    def test_detect_unknown_base_model(self, manager):
        cfg = {}
        result = manager._detect_base_model_from_adapter_config(cfg)
        assert result == LoRABaseModel.UNKNOWN

    def test_detect_image_use_case_from_diffusers(self, manager):
        cfg = {"auto_mapping": {"parent_library": "diffusers", "base_model_class": ""}}
        result = manager._detect_use_case_from_adapter_config(cfg)
        assert result == LoRAUseCase.IMAGE

    def test_detect_text_use_case_from_transformers(self, manager):
        cfg = {"auto_mapping": {"parent_library": "transformers", "base_model_class": ""}}
        result = manager._detect_use_case_from_adapter_config(cfg)
        assert result == LoRAUseCase.TEXT

    def test_detect_image_use_case_from_base_model_path(self, manager):
        cfg = {"base_model_name_or_path": "runwayml/stable-diffusion-v1-5"}
        result = manager._detect_use_case_from_adapter_config(cfg)
        assert result == LoRAUseCase.IMAGE


# ---------------------------------------------------------------------------
# LoRAManager – get_compatible_loras
# ---------------------------------------------------------------------------

class TestGetCompatibleLoras:
    def _add_image_lora(self, manager, name, base=LoRABaseModel.SD15):
        lora = LoRAInfo(name=name, path="/p", use_case=LoRAUseCase.IMAGE, base_model=base)
        manager.image_loras[name] = lora

    def test_returns_all_when_no_model_type(self, manager):
        self._add_image_lora(manager, "LoraA")
        self._add_image_lora(manager, "LoraB")
        result = manager.get_compatible_loras(LoRAUseCase.IMAGE)
        assert len(result) == 2

    def test_filters_by_base_model_type(self, manager):
        self._add_image_lora(manager, "SD15Lora", LoRABaseModel.SD15)
        self._add_image_lora(manager, "SDXLLora", LoRABaseModel.SDXL)
        result = manager.get_compatible_loras(LoRAUseCase.IMAGE, loaded_model_type="sd15")
        names = [l.name for l in result]
        assert "SD15Lora" in names

    def test_includes_unknown_base_model(self, manager):
        self._add_image_lora(manager, "UnknownLora", LoRABaseModel.UNKNOWN)
        result = manager.get_compatible_loras(LoRAUseCase.IMAGE, loaded_model_type="sd15")
        names = [l.name for l in result]
        assert "UnknownLora" in names

    def test_text_loras(self, manager):
        lora = LoRAInfo(name="TextLora", path="/p", use_case=LoRAUseCase.TEXT, base_model=LoRABaseModel.MISTRAL)
        manager.text_loras["TextLora"] = lora
        result = manager.get_compatible_loras(LoRAUseCase.TEXT)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# LoRAManager – get_lora_choices
# ---------------------------------------------------------------------------

class TestGetLoraChoices:
    def test_none_is_always_first(self, manager):
        lora = LoRAInfo(name="X", path="/p", use_case=LoRAUseCase.IMAGE, base_model=LoRABaseModel.SD15)
        manager.image_loras["X"] = lora
        choices = manager.get_lora_choices(LoRAUseCase.IMAGE)
        assert choices[0] == "None"

    def test_includes_lora_names(self, manager):
        lora = LoRAInfo(name="MyLora", path="/p", use_case=LoRAUseCase.IMAGE, base_model=LoRABaseModel.SD15)
        manager.image_loras["MyLora"] = lora
        choices = manager.get_lora_choices(LoRAUseCase.IMAGE)
        assert "MyLora" in choices


# ---------------------------------------------------------------------------
# LoRAManager – get_lora_info
# ---------------------------------------------------------------------------

class TestGetLoraInfo:
    def test_finds_image_lora(self, manager):
        lora = LoRAInfo(name="Img", path="/p", use_case=LoRAUseCase.IMAGE, base_model=LoRABaseModel.SD15)
        manager.image_loras["Img"] = lora
        assert manager.get_lora_info("Img") is lora

    def test_finds_text_lora(self, manager):
        lora = LoRAInfo(name="Txt", path="/p", use_case=LoRAUseCase.TEXT, base_model=LoRABaseModel.MISTRAL)
        manager.text_loras["Txt"] = lora
        assert manager.get_lora_info("Txt") is lora

    def test_returns_none_for_unknown(self, manager):
        assert manager.get_lora_info("no_such_lora") is None


# ---------------------------------------------------------------------------
# LoRAManager – get_all_loras_summary
# ---------------------------------------------------------------------------

class TestGetAllLorasSummary:
    def test_no_loras_message(self, manager):
        summary = manager.get_all_loras_summary()
        assert "No LoRAs" in summary

    def test_shows_lora_names(self, manager):
        lora = LoRAInfo(name="Cool", path="/p", use_case=LoRAUseCase.IMAGE, base_model=LoRABaseModel.SD15)
        manager.image_loras["Cool"] = lora
        summary = manager.get_all_loras_summary()
        assert "Cool" in summary


# ---------------------------------------------------------------------------
# LoRAManager – update_lora_metadata
# ---------------------------------------------------------------------------

class TestUpdateLoraMetadata:
    def test_update_nonexistent_returns_false(self, manager):
        assert manager.update_lora_metadata("ghost", {"description": "x"}) is False

    def test_update_in_memory(self, tmp_path, manager):
        lora_dir = tmp_path / "loras" / "update_lora"
        lora_dir.mkdir(parents=True)
        lora = LoRAInfo(
            name="update_lora", path=str(lora_dir),
            use_case=LoRAUseCase.IMAGE, base_model=LoRABaseModel.SD15
        )
        manager.image_loras["update_lora"] = lora
        result = manager.update_lora_metadata("update_lora", {"description": "Updated!"})
        assert result is True
        assert lora.description == "Updated!"
