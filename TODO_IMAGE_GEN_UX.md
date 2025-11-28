# Image Generator UX Improvements TODO

## 1. Model Presets Config File ✅ DONE
- [x] Create `config/image_model_presets.yaml` with per-model settings
- [x] Load presets at startup
- [x] Auto-update sliders when model changes
- [x] Update AVAILABLE_IMAGE_MODELS to use config file

## 2. Button State Management ✅ DONE
- [x] Hide Stop button when generation is not running
- [x] Disable Generate button when generation is running
- [x] Hide Unload button if selected model is not loaded
- [x] Disable Generate button when no model is loaded or loading

## 3. Image Metadata Storage ✅ DONE
- [x] Save PNG metadata with each generated image
- [x] Store: prompt, negative_prompt, model, steps, guidance, seed, dimensions, lora, generation_time
- [x] Update generate_image to track and save metadata

## 4. Gallery Improvements ✅ DONE
- [x] Show image metadata when viewing in gallery
- [x] Add "Load Settings" button to reuse settings from image
- [x] Multi-select with checkboxes
- [x] Bulk delete and download (ZIP)
- [x] Better selection UI with visual feedback

## 5. Prompt Enhancement ✅ DONE
- [x] Add Flan-T5-Small model loader (CPU-only, ~300MB)
- [x] Lazy load on first use (not at startup)
- [x] Quick enhance (templates only - instant)
- [x] AI enhance (Flan-T5 powered)
- [x] Auto-detect style from prompt content
- [x] Live CLIP token counter (77 token limit)

## 6. LoRA Training System ✅ DONE
- [x] Dataset management (create, upload, caption images)
- [x] Training UI with progress tracking
- [x] Multiple crop modes (resize_pad, smart, center, top, stretch)
- [x] Resume training from existing LoRA
- [x] LoRA selector in image generator
- [x] My LoRAs management tab

## 7. Prompt Library ✅ DONE
- [x] Save prompts with all settings
- [x] Load prompts back to generator
- [x] Delete saved prompts
- [x] Preview prompt details

---

## Future Improvements (Not Started)

### Image Generation
- [ ] Img2Img support
- [ ] Inpainting
- [ ] ControlNet integration
- [ ] Batch generation (queue multiple prompts)
- [ ] Upscaling (Real-ESRGAN)

### LoRA Training
- [ ] Auto-captioning with BLIP (partially implemented)
- [ ] Training preview images
- [ ] Learning rate finder
- [ ] Multi-concept training

### UI/UX
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts
- [ ] Drag & drop image upload
- [ ] Side-by-side comparison view

### Performance
- [ ] Model quantization (8-bit, 4-bit)
- [ ] TensorRT optimization
- [ ] Batch inference
