# KVGenius

AI image generation desktop app — a lightweight Electron front end over a
locally-running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server.
KVGenius doesn't load any models itself; it submits curated workflow
templates to your ComfyUI instance over HTTP and displays the results.

**Status:** rewrite in progress. The previous Flet/Python implementation is
preserved on the [`legacy-flet-app`](https://github.com/gerp93/KVGenius/tree/legacy-flet-app)
branch for reference; `main` now holds the new Electron version going
forward.

Follows [gerp93/KVG_Standards](https://github.com/gerp93/KVG_Standards) for
theming, licensing, release/CI, update-check, and the rest of the org's app
conventions.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
