# LoRA & Checkpoint System TODO

## Overview
Expand the model loading system to support checkpoints, multiple simultaneous LoRAs, and a dedicated management interface for all LoRA types (image, text, CAH).

---

## Phase 1: Checkpoint Support
- [ ] **Detect checkpoint vs diffusers format**
  - Check for single `.safetensors`/`.ckpt` file vs folder with `model_index.json`
  - Add `model_type` field to image presets config: `"diffusers"` | `"checkpoint"`

- [ ] **Implement checkpoint loading**
  - Use `StableDiffusionPipeline.from_single_file()` for checkpoints
  - Use `StableDiffusionPipeline.from_pretrained()` for diffusers (current)
  - Support both SD 1.5 and SDXL checkpoint formats

- [ ] **Add local checkpoints to config**
  - New section in `image_model_presets.yaml` for local checkpoints
  - Fields: `path`, `name`, `base_type` (sd15/sdxl), `vram`, `default_settings`

---

## Phase 2: Multiple LoRAs Stacking
- [ ] **Update LoRA loading for multi-adapter support**
  ```python
  pipe.load_lora_weights("lora1.safetensors", adapter_name="lora1")
  pipe.load_lora_weights("lora2.safetensors", adapter_name="lora2")
  pipe.set_adapters(["lora1", "lora2"], adapter_weights=[0.7, 0.5])
  ```

- [ ] **UI for multiple LoRA selection**
  - Multi-select dropdown or checkboxes for available LoRAs
  - Individual weight slider per selected LoRA (0.0 - 1.5)
  - Show combined trigger words from all selected LoRAs

- [ ] **LoRA compatibility checking**
  - Validate LoRA base model matches loaded model (SD 1.5 vs SDXL)
  - Warn if mixing incompatible LoRAs

---

## Phase 3: Dedicated LoRA Management Page
- [ ] **Create new Gradio tab: "LoRA Manager"**
  - Three sections with accordions:
    1. **Image LoRAs** - For Stable Diffusion models
    2. **Text Model LoRAs** - For chat/roleplay (PEFT adapters)
    3. **CAH LoRAs** - Specialized for card generation themes

- [ ] **LoRA browser per section**
  - Grid view with preview thumbnails
  - Name, description, trigger words display
  - Base model compatibility badge
  - Enable/disable toggle
  - Weight slider

- [ ] **LoRA organization**
  - Categories/tags for filtering
  - Search functionality
  - Favorites/pinned LoRAs

---

## Phase 4: Text Model LoRA Support
- [ ] **Integrate PEFT library for text models**
  ```python
  from peft import PeftModel
  model = AutoModelForCausalLM.from_pretrained(base_model)
  model = PeftModel.from_pretrained(model, lora_path)
  ```

- [ ] **LoRA switching for personas**
  - Load different LoRAs for different chat personas
  - Hot-swap LoRAs without reloading base model

- [ ] **CAH-specific LoRAs**
  - Train/use LoRAs for specific card themes (dark humor, pop culture, etc.)
  - Apply LoRA before card generation

---

## Phase 5: Config & Metadata System
- [ ] **Create `config/lora_presets.yaml`**
  ```yaml
  image_loras:
    - name: "Realspice"
      path: "data/lora_models/Realspice"
      trigger_words: ["realspice style"]
      base_model: "sd15"  # sd15, sdxl, or specific model ID
      compatible_models: ["Dreamshaper 8", "Realistic Vision V6", "epiCRealism"]
      use_case: "image"  # image, text, cah
      default_weight: 0.8
      category: "style"
      preview: "preview.png"
  
  text_loras:
    - name: "Roleplay Enhancer"
      path: "data/lora_models/roleplay"
      base_model: "mistral"  # mistral, llama, etc.
      compatible_models: ["Nous-Hermes-2-Mistral-7B", "Dolphin-2.6-Mistral-7B"]
      use_case: "text"
      category: "persona"
  
  cah_loras:
    - name: "Dark Humor"
      path: "data/lora_models/cah_dark"
      base_model: "llama"
      compatible_models: ["Dolphin-2.9-Llama3-8B"]
      use_case: "cah"
      category: "theme"
  ```

- [ ] **LoRA metadata fields (required)**
  - `use_case`: "image" | "text" | "cah" - determines which tab/section shows the LoRA
  - `base_model`: Architecture type (sd15, sdxl, mistral, llama, etc.)
  - `compatible_models`: List of specific model names this LoRA works with
  - Auto-filter dropdowns based on currently loaded model

- [ ] **Per-LoRA metadata files**
  - `lora_config.json` in each LoRA folder
  - Fields: trigger_words, base_model, compatible_models, use_case, author, version, description
  - Auto-detect and parse on startup

- [ ] **Preview image system**
  - Look for `preview.png`/`preview.jpg` in LoRA folder
  - Display in LoRA manager grid
  - Generate previews if missing (optional)

---

## Phase 6: UI Refactoring - Move LoRA Controls
- [ ] **Remove LoRA section from Image Generation tab**
  - Delete current LoRA dropdown/controls from image gen accordion
  - Keep only a "Selected LoRAs" display showing active LoRAs + weights

- [ ] **Add LoRA quick-select in each generation tab**
  - Image Gen: Small filtered dropdown showing only compatible image LoRAs
  - Chat: Small filtered dropdown showing only compatible text LoRAs  
  - CAH Generator: Small filtered dropdown showing only compatible CAH LoRAs
  - Link to "Manage LoRAs" tab for full control

- [ ] **Auto-filter LoRA dropdowns by loaded model**
  - When user loads "Dreamshaper 8", only show SD 1.5 compatible LoRAs
  - When user loads "Dolphin-2.9-Llama3-8B", only show Llama-compatible text LoRAs
  - Gray out / hide incompatible LoRAs with tooltip explaining why

---

## Technical Notes

### Checkpoint Loading
```python
# Detect format
def get_model_type(path):
    if os.path.isfile(path) and path.endswith(('.safetensors', '.ckpt')):
        return 'checkpoint'
    elif os.path.isdir(path) and os.path.exists(os.path.join(path, 'model_index.json')):
        return 'diffusers'
    return None

# Load accordingly
if model_type == 'checkpoint':
    pipe = StableDiffusionPipeline.from_single_file(path, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
```

### Multiple LoRAs with Diffusers
```python
# Load multiple LoRAs
pipe.load_lora_weights("lora1.safetensors", adapter_name="style")
pipe.load_lora_weights("lora2.safetensors", adapter_name="character")

# Combine with weights
pipe.set_adapters(["style", "character"], adapter_weights=[0.7, 0.5])

# Generate with combined LoRAs
image = pipe(prompt, ...).images[0]

# Disable all LoRAs
pipe.disable_lora()
```

### Text Model LoRAs (PEFT)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("base_model_id")
tokenizer = AutoTokenizer.from_pretrained("base_model_id")

# Apply LoRA
model = PeftModel.from_pretrained(base_model, "path/to/lora")

# Switch LoRA (if multiple loaded)
model.set_adapter("persona_lora")
```

---

## File Structure After Implementation
```
data/
  lora_models/
    image/
      Realspice/
        realspice.safetensors
        lora_config.json
        preview.png
      StarWars/
        ...
    text/
      roleplay_enhancer/
        adapter_config.json
        adapter_model.safetensors
        lora_config.json
    cah/
      dark_humor/
        ...
  checkpoints/
    epicRealism.safetensors
    ...
config/
  lora_presets.yaml
  image_model_presets.yaml  # Add checkpoint entries
```

---

## Priority Order
1. ✅ Checkpoint support (quick win, enables more models)
2. ✅ Multiple LoRAs stacking (high user value)
3. ✅ LoRA config system (foundation for management)
4. ✅ LoRA management page (UX improvement)
5. ✅ Text model LoRAs (advanced feature)
