# Version bump

Auto Release (`.github/workflows/auto-release.yml`) always does at least a
patch bump on every push to `main` — it doesn't care what changed. So to
force a release with no real code change (e.g. to pick up an updated
KVG_Standards reusable workflow), edit this file instead of pushing an
empty commit. Add a one-line entry below with the date and why, so the
commit shows a real diff instead of nothing.

- 2026-08-07 — created this file (KVG_Standards compliance audit)
