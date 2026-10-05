# CLAUDE.md — KVGenius

## Standards

This repo follows [gerp93/KVG_Standards](https://github.com/gerp93/KVG_Standards)
for theming, licensing, release/CI, update-check, database location, and the
Electron application menu. Consult that repo (or its `app-standards` Claude
Code skill) before diverging from an existing pattern — a one-off
implementation here is exactly the kind of drift it exists to prevent.

## Architecture

- `src/main/` — Electron main process: window/menu setup (`main.ts`,
  `menu.ts`), the ComfyUI HTTP client (`comfyui.ts`), SQLite via `node:sqlite`
  (`db.ts`), and the config/db-location pattern (`dbLocation.ts`).
- `src/preload/` — `contextBridge` API surface exposed to the renderer as
  `window.kvgenius` (see `src/shared/types.ts`'s `KVGeniusAPI` for the
  contract both sides implement).
- `src/renderer/` — React UI (`pages/Generate.tsx`, `pages/Library.tsx`).
- `src/shared/` — types used by both main and renderer.
- `src/mcp/` — the MCP stdio shim (`shim.ts`, protocol in `mcpProtocol.ts`). Runs as its own
  process (Node, or the app binary with `ELECTRON_RUN_AS_NODE=1`), holds no state, and forwards to
  the running app's local API. See `docs/mcp-plan.md`.
- `src/main/templates/` — ComfyUI workflow templates in API format (not the
  visual "workflow format" you'd drag into ComfyUI's own editor — see the
  session history for why that distinction matters). One template per model
  family (`z-image-turbo.json` for image mode, `wan22-i2v.json` for video
  mode); `src/main/comfyui.ts` has a separate node-ID map per family (e.g.
  `WAN22_I2V_NODE_MAP`) recording which fields get patched at generation
  time, and these must stay in sync with their template JSON. `wan22-i2v`
  patches only the fields a user actually needs (prompt, width/height/length,
  seed, source image) plus one boolean, the 4-step-LoRA switch (`129:131`),
  which backs the Fast/High video quality option (`src/shared/videoQuality.ts`;
  stored as steps/cfg on the record, so cfg > 1 means High). The step/CFG
  primitives it selects between, including the CFG=1 feeding the second
  KSampler pass, are left as the template author set them - matching the
  curated-field philosophy below.
- `src/shared/types.ts`'s `FAMILY_KIND` map says whether a family produces
  an image or a video - the renderer uses it to decide `<img>` vs `<video>`
  for a given record, without needing a separate DB column for it.

- Generation goes through a persisted job queue in the main process (`jobQueue.ts`,
  `jobStore.ts`; the runner is `generationService.ts`). The UI's `generate` IPC and outside
  clients (`apiService.ts` over the loopback API in `localApi.ts`) submit to the same queue, so
  they share one line for the GPU. Tool definitions live in `src/shared/tools.ts`; add a tool
  there and in `ApiService.callTool`. ffmpeg work (`mediaTools.ts`, `assembly.ts`) is separate
  from the queue. `npm test` runs the `*.test.ts` files (compiled by `tsconfig.test.json`).

## Key design decisions

- **`node:sqlite`, not `sql.js`** — KVG_Standards documents `sql.js`'s
  whole-file-overwrite-on-save as the root cause of real data-loss bugs in
  several sibling repos. New Electron apps in this org should start with
  `node:sqlite` (RolePlaymate is the other reference implementation).
- **A curated template per model family, not arbitrary ComfyUI workflow
  import** — see the session history for the full reasoning; short version:
  hand-built templates avoid depending on custom ComfyUI node packages that
  may not be installed, and keep the exposed field set predictable per mode.
- Generated images are served to the renderer via a custom `kvimage://`
  protocol (not raw `file://`), matching RolePlaymate's `rpimage://` pattern
  — `file://` subresources don't load in the dev window (which runs against
  `http://localhost:5173`), and the custom scheme behaves identically in dev
  and packaged builds.
- **No separate "saved prompt" record.** Library > Prompts is the generations
  with `pinned_at` set - a picture chosen as the example of a look, whose prompt
  is just that generation's own. An image-less prompt, a name or tags were
  deliberately dropped (the table was migrated into pins by `migrateSavedPrompts`
  in `db.ts`, with any prompt that had no matching image written to
  `saved-prompts-unpinned.txt` beside the database). Deleting a pinned
  generation removes the pin with it.
- **A video keeps its own copy of its source image** (`source_image_path`, files in
  `output/sources`, named by content hash so one picture is stored once - see
  `sourceImages.ts`). The original is only ever uploaded to ComfyUI and may be anywhere
  on disk or move/vanish inside the Library, so without the copy Re-rack could not re-run
  a video in place. The copy is deleted with the last video that uses it; videos made
  before this have none and need a source chosen again.
