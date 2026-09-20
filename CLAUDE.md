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
- `src/main/templates/` — ComfyUI workflow templates in API format (not the
  visual "workflow format" you'd drag into ComfyUI's own editor — see the
  session history for why that distinction matters). One template per model
  family; `src/main/comfyui.ts`'s `NODE_MAP` records which node IDs get
  patched at generation time and must stay in sync with the template JSON.

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
