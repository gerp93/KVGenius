# Image Generator UX Improvements TODO

## 1. Model Presets Config File
- [ ] Create `config/image_model_presets.yaml` with per-model settings
- [ ] Load presets at startup
- [ ] Auto-update sliders when model changes
- [ ] Update AVAILABLE_IMAGE_MODELS to use config file

## 2. Button State Management
- [ ] Hide Stop button when generation is not running
- [ ] Disable Generate button when generation is running
- [ ] Hide Unload button if selected model is not loaded
- [ ] Disable Generate button when no model is loaded or loading

## 3. Image Metadata Storage
- [ ] Save JSON metadata alongside each generated image
- [ ] Store: prompt, negative_prompt, model, steps, guidance, seed, dimensions, generation_time
- [ ] Update generate_image to track and save metadata

## 4. Gallery Improvements
- [ ] Show image metadata when viewing in gallery
- [ ] Add "Copy Settings" button to reuse settings
- [ ] Add "Copy Prompt" button
- [ ] Better selection UI with visual feedback

## 5. Prompt Improvement AI (CPU-based)
- [ ] Add flan-t5-base model loader (CPU-only)
- [ ] Pre-load at startup
- [ ] Add "✨ Enhance Prompt" button
- [ ] Preview enhanced prompt before applying
