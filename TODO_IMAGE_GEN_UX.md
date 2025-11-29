# KVGenius - Feature Roadmap

## Completed ✅

### Phase 1: Core Image Generation
- [x] Model presets config file with per-model settings
- [x] Button state management (show/hide based on state)
- [x] Image metadata storage (PNG with full settings)
- [x] Gallery with metadata viewing and bulk operations
- [x] Prompt enhancement (Quick + AI-powered with Flan-T5)
- [x] LoRA training system (dataset, train, manage)
- [x] Prompt library (save/load prompts)
- [x] Live CLIP token counter (77 token limit)

### Phase 2: Project Organization
- [x] Reorganized folder structure (data/, scripts/, src/)
- [x] Model-agnostic configuration (user configs git-ignored)
- [x] Chat model presets (`config/chat_model_presets.yaml`)
- [x] Image model presets (`config/image_model_presets.yaml`)
- [x] Consistent config pattern for chat and image models

### Phase 3: UI/UX Improvements
- [x] Restructured Chat tab with sub-tabs (Conversation, Characters, Personas)
- [x] Image Generator has sub-tabs (Generate, Gallery, Prompt Library, LoRA Training)
- [x] Consistent layouts between Chat and Image Generator
- [x] Better progress bar with step-by-step updates
- [x] LoRA import from .safetensors files
- [x] Auto-refresh LoRA dropdown after import/delete

---

## In Progress 🔄

### Character Appearance System
- [ ] Link chat characters to visual appearance descriptions
- [ ] Store appearance prompts (hair, eyes, clothing, style)
- [ ] Optional LoRA association per character
- [ ] Auto-fill image prompts when character selected
- [ ] Character portrait generation workflow

---

## Planned 📋

### Image Generation
- [ ] Img2Img support
- [ ] Inpainting
- [ ] ControlNet integration
- [ ] Batch generation (queue multiple prompts)
- [ ] Upscaling (Real-ESRGAN)

### LoRA Training
- [ ] Auto-captioning with BLIP
- [ ] Training preview images
- [ ] Learning rate finder
- [ ] Multi-concept training
- [ ] Chat model LoRA training (personality fine-tuning)

### UI/UX
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts
- [ ] Drag & drop improvements
- [ ] Side-by-side comparison view
- [ ] Move LoRA Training to top-level tab (when chat LoRAs added)

### Performance
- [ ] Model quantization (8-bit, 4-bit)
- [ ] TensorRT optimization
- [ ] Batch inference

---

## Future Ideas 💡

### Character Appearance System (Detailed Plan)
```yaml
# Example character with linked appearance
name: "Luna"
personality: "A mysterious space explorer..."
appearance:
  base_prompt: "young woman, silver hair, violet eyes"
  style: "sci-fi, futuristic"
  clothing: "black flight suit, utility belt"
  lora: "Luna_v1"
  trigger_word: "luna_char"
```

**Workflow:**
1. Create character in Chat → Characters
2. Add visual appearance details
3. Generate reference images
4. Train LoRA on best images
5. Link LoRA to character
6. Auto-apply when generating images of that character

### Chat Model LoRA Training
Fine-tune chat models on character-specific conversations:
1. Export chat history with character
2. Format as training data
3. Train LoRA on personality/style
4. Apply to base chat model
5. More consistent character responses
