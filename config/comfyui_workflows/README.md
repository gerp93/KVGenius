# ComfyUI workflow templates

These JSON files are ComfyUI API-format workflow graphs, loaded and patched at
runtime by `core/comfyui_client.py`. They assume a single "all-in-one"
checkpoint file loaded via `CheckpointLoaderSimple` (the common case for
SD1.5/SDXL, and for Flux checkpoints that bundle UNet+CLIP+VAE together).

`__CKPT_NAME__`, `__PROMPT__`, and `__NEGATIVE_PROMPT__` are string
placeholders substituted by `comfyui_client.patch_workflow()`. Numeric fields
(steps/cfg/width/height/seed) are overwritten directly by node ID, hardcoded
in `comfyui_client.NODE_MAP` to match the node IDs used in these files (`3`
KSampler, `4` CheckpointLoaderSimple, `5` EmptyLatentImage, `6`/`7`
CLIPTextEncode positive/negative, `8` VAEDecode, `9` SaveImage). If you edit
these graphs, keep those IDs or update `NODE_MAP` to match.

When a LoRA is requested, `comfyui_client.patch_workflow()` inserts a
`LoraLoader` node (id `10`) between the checkpoint loader and everything that
consumes its model/clip outputs.

**Flux caveat:** if your ComfyUI setup loads Flux via separate `UNETLoader` +
`DualCLIPLoader` + `VAELoader` nodes instead of one merged checkpoint, this
template won't match your setup — edit `flux.json` (and `NODE_MAP` if you
change node IDs) to match your actual Flux graph. `cfg`/`sampler`/`scheduler`
defaults here are a reasonable starting point, not tuned guidance from a
`FluxGuidance` node.
