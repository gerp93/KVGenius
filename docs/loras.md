# LoRAs (species, characters, looks)

Describing something the base model has never seen ("a Twi'lek") rarely works: the model does not know the
concept. A **LoRA** is a small add-on file that teaches it. KVGenius carries a LoRA on an **element** (Styles >
Elements), so picking the element in Generate applies everything it needs: its wording (which holds the LoRA's
trigger word) and the LoRA file itself.

## Using one

1. Put the LoRA file in ComfyUI's `models/loras` folder yourself. KVGenius does not download model files.
2. Styles > Elements > New (or edit one). Put the trigger word(s) in the element text. Under **LoRA** pick the
   file (the list comes from ComfyUI; if ComfyUI is not running, type the file name instead) and a strength
   (default 1, allowed -2 to 2).
3. In Generate, use the **Flux dev** model and add the element. A line under the elements says which LoRA loads.
   Up to 3 LoRAs chain, in the order the elements were added.
4. The result's details panel lists the LoRAs it was made with, and Library > Output treats a different LoRA or
   strength as a different picture (the repeat check).

**Only Flux dev can load a LoRA for now.** With any other model (Z-Image, Qwen-Image, HiDream, Ovis) the element's
words are still added but the LoRA is left out, and Generate (and the API reply, as `notes`) says so; it is never
an error. Adding a family later is small: list it in `LORA_FAMILIES` (`src/shared/loras.ts`) and give it an entry
in `LORA_NODE_MAP` (`src/main/loraPatch.ts`, the node that loads the model and the inputs that consume it).

How it works: the shipped templates contain no LoRA node (ComfyUI rejects a `lora_name` it does not have, so a
placeholder would break every normal run). When a job carries LoRAs, `applyLoras` adds one core
`LoraLoaderModelOnly` node per LoRA between the model loader and the sampler. The job (and the Library record)
keeps the file names and strengths, so a result can be reproduced.

Over MCP, `list_styles` shows an element's LoRA file name and strength, and `generate_image` with
`family: "flux-dev"` and that element in `elements` loads it.

## Example: Twi'lek (Star Wars)

FullFantasyLady's **Ancestries Compendium** on Civitai (a Flux.1 dev LoRA; model 1893473), **Twi'lek Mk.1**
version.

Checked from the downloaded file's header (`FullFantasyLady-Twilek-Mk1.safetensors`, about 18 MB):

- It is a Flux.1 dev LoRA (`modelspec.architecture` = `flux-1-dev/lora`), in kohya format with UNet keys only
  (no text-encoder keys), so the model-only loader KVGenius uses is the right one. Network dim 2, alpha 16, trained
  on 141 images.
- The trigger word is exactly `ffl-twi'lek-mk.1`, the only caption tag it was trained with. Put it in the element
  text, e.g. `ffl-twi'lek-mk.1, a Twi'lek woman`.

> **Still unverified.** These come from the model page's summary, not from the file or from running it (no
> generation with this LoRA has been seen in KVGenius). Treat them as starting points and adjust by eye.

- Flux guidance (the CFG box for Flux) about 2.5 to 3.5.
- LoRA strength about 0.5 to 1.0.
- About 40 steps.

## LoRAs that are not Flux (checked, not usable yet)

Two more files were looked at by reading only their safetensors headers (`__metadata__` and tensor names, no
weights). **Neither is a Flux LoRA, so neither works with the model-only loader KVGenius uses, and neither was
copied into ComfyUI's `models/loras`.** No generation with either has been run or seen.

| File | Base model | Text-encoder weights | Trigger word |
| --- | --- | --- | --- |
| `Twilek.safetensors` (about 218 MB) | **SDXL** (`modelspec.architecture` = `stable-diffusion-xl-v1-base/lora`, `ss_base_model_version` = `sdxl_base_v1-0`), trained on `luminaversePonyXL_v10`, a Pony-family SDXL checkpoint | Yes: `lora_te1_*` and `lora_te2_*` (792 keys) | `twi'lek` (in all 405 training captions) |
| `Environment - StarWars.safetensors` (about 144 MB) | **Stable Diffusion 1.x** (inferred: kohya keys `lora_unet_down_blocks_*` with one `lora_te_*` encoder and 768-wide attention; the file has no `modelspec` or `ss_base_model_version` entry) | Yes: `lora_te_*` (216 keys) | `starwars` (in all 255 training captions) |

Other facts read from the headers: `Twilek` is kohya LoRA dim 32, alpha 16, 405 images, 1024x1024, trained 2024-06;
its captions are booru-style tags (it is an adult-content LoRA, like the Pony models it was trained on). `starwars`
is dim 128, alpha 128, 255 images at 512x512 with clip skip 2; its captions are mostly scenery tags (`scenery`,
`science_fiction`, `spacecraft`, `outdoors`, `no_humans`), so it is a landscape/vehicle look rather than a character.

Why they cannot be added like the Flux one: `applyLoras` adds `LoraLoaderModelOnly` nodes, which patch the diffusion
model only and silently drop the `lora_te*` weights. Both of these files carry text-encoder weights, and above all
KVGenius has no SDXL or SD 1.5 family at all (no checkpoint loader, no CLIP loader, no template), so there is nothing
for them to load into. Adding one needs: a template per base model (SDXL and SD 1.5 are different graphs; a Pony
checkpoint would be a model file the user supplies), a family entry and node map, and a second LoRA path that uses
core `LoraLoader` (model **and** clip) so the text-encoder weights count. That is a new family plus a LoRA-loader
variant, not a small change. Until then, an element carrying either file would only add its words.
