# KVGenius

A lightweight Electron desktop app for AI image generation — a thin wrapper
over a locally-running [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
server. KVGenius doesn't load any models or run any inference itself: it
submits a curated ComfyUI workflow template over HTTP, waits for the result,
and saves the generated image alongside the prompt and settings that
produced it.

**Requires a running ComfyUI instance** (default `http://localhost:8000`,
ComfyUI Desktop's own default - the standalone ComfyUI server defaults to
`8188` instead, which is configurable in Settings) with the model files the
active template expects already installed. See `src/main/templates/` for
the workflow templates KVGenius ships with.

**Status:** early rewrite. Currently supports one mode (plain text-to-image)
against one model template (Z Image Turbo). Image-to-image, inpainting, and
additional model families are planned — see [TODO.md](TODO.md).

The previous Flet/Python implementation of this app is preserved on the
[`legacy-flet-app`](https://github.com/gerp93/KVGenius/tree/legacy-flet-app)
branch for reference.

## Development

```bash
npm install
npm run dev
```

## Building

```bash
npm run build
npm run package
```

## Standards

Follows [gerp93/KVG_Standards](https://github.com/gerp93/KVG_Standards) for
theming (VisualAssault, vendored in `src/renderer/themes.css`), licensing,
release/CI (`.github/workflows/auto-release.yml` /
`.github/workflows/cut-release.yml` → `release-electron.yml`), update-check
(`electron-updater`), the application menu (`src/main/menu.ts`), and
database location (`src/main/dbLocation.ts`).

## License

AGPL-3.0 — see [LICENSE](LICENSE).
