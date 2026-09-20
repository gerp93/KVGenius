# TODO

This app's own backlog of future features and fixes — not a KVG_Standards
compliance checklist (see [KVG_Standards](https://github.com/gerp93/KVG_Standards)
and this repo's `REPO_SCOPE.md` entry for that). Just what's not built yet.

## Features

- Image-to-image and inpainting modes (Generate/txt2img is the only mode so far)
- Real negative prompt for the Z Image Turbo template, gated behind raising
  CFG above 1 (currently zeroed out — see `src/main/comfyui.ts`'s node map
  and the template's `ConditioningZeroOut` node). Needs an actual
  side-by-side quality comparison before deciding whether to ship it as a
  default, not just wiring it in blind.
- Support for additional image model families beyond Z Image Turbo
  (SD1.5-style, SDXL-style, Flux-style templates), each with its own
  curated field set

## Needs real-world verification

- Video mode (Wan2.2 I2V, `wan22-i2v` family) was verified via a mock
  ComfyUI server (confirmed correct node patching, image upload, output
  extraction, DB storage, and Library/recall rendering) but never run
  against a real ComfyUI instance with the actual Wan2.2 models - no GPU
  or multi-GB model files available in the dev/build sandbox. Needs a real
  end-to-end run to confirm actual video generation works, and to confirm
  ComfyUI's native SaveVideo node really does output under a `videos` key
  in `/history` (assumed, not confirmed against real ComfyUI source).
- Real step-by-step generation progress (currently an indeterminate
  spinner) - would need ComfyUI's WebSocket API (`/ws`, `progress`
  messages) piped through an IPC event channel to the renderer.

## Fixes

- `assets/logo.png` doesn't exist yet — needs real artwork before
  `scripts/generate-icons.js` and the packaged-binary icon can be wired in
  (deliberately not fabricated, per KVG_Standards' logo & branding standard)
- No startup database-recovery flow yet (a missing/locked db path currently
  just fails startup via the generic error dialog) — see RolePlaymate's
  `openDatabaseWithRecovery` in `src/main/main.ts` for the reference pattern
  if this becomes a real problem
