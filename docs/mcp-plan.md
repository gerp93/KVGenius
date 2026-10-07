# MCP integration plan

Status: **phases 1-4 implemented** (see "As built" below); phase 5 is future
work. Written from a design conversation, then built.

## Goal

Let an MCP client (Claude Desktop first) drive KVGenius's existing generation
engine, so a user can say things like:

> Take the images in `~/shoot/` and this script. Write an image-to-video prompt
> for each, generate the clips with KVGenius, then stitch them into a music
> video with `~/song.wav` underneath.

KVGenius stays a small, reliable set of **primitives**. Composition ("any
combination of workflows") is the client's job: the client chains tool calls.
We do not build a workflow engine into the app.

## Non-goals (v1)

- Headless operation. The app must be running (and ComfyUI reachable).
- Client-specific behaviour. Claude Desktop is the first client, not the design
  target; nothing in tool design may depend on it.
- Beat detection / interpreting the music. The audio is a backing track only.
- Attaching images via the chat UI. Inputs come from local folder/file paths.
- Arbitrary ComfyUI workflow import (see CLAUDE.md: curated templates).

## Architecture

```
MCP client (Claude Desktop)
   │ stdio
   ▼
stdio shim  ── forwards ──►  local HTTP API (127.0.0.1, token)   ◄── future clients
                                   │
                              service layer  (transport-agnostic, typed ops)
                                   │
        job queue ── comfyui.ts ── db.ts (node:sqlite) ── ffmpeg
                                   ▲
                              Electron UI (uses the same service layer)
```

- **Service layer** (new, `src/main/service/` or similar): typed operations
  such as `submitJob`, `listJobs`, `importFolder`, `assemble`. Input/output
  schemas are defined once in `src/shared/` and reused by the IPC handlers, the
  HTTP API and the MCP tool definitions. MCP and HTTP are thin adapters.
- **Why the app owns everything:** Claude Desktop launches MCP servers as stdio
  subprocesses, but the Electron app is already the single owner of the ComfyUI
  connection, the job queue and SQLite. A second process touching the DB or
  ComfyUI would fight it. So the shim is dumb: it speaks MCP on stdio and
  forwards to the app's local HTTP API. The shim can be the same binary started
  with a flag (e.g. `--mcp`) or a tiny separate entry point.
- **Local HTTP API:** binds `127.0.0.1` only, requires a bearer token (generated
  at first enable, stored in the app config), rejects requests whose `Origin`
  header is set (DNS-rebinding defence). Enabled via an in-app toggle; off by
  default.
- **Client-agnostic:** results use standard MCP content (image blocks inline,
  IDs and file paths for everything else). Adding a client later = adding an
  adapter, not changing tools. Packaging the shim as a Claude Desktop
  extension for one-click install is a later, additive step.

## Prerequisite: real job queue

**Phase 1 status: implemented** (`jobStore.ts`, `jobQueue.ts`,
`generationService.ts`, `src/shared/jobs.ts`; tests via `npm test`). The
Generate page already had a queue, but it lived in the renderer, invisible to
anything else, and `comfyui.ts` tracks a single in-flight prompt. Now the
`generate` IPC handler submits to a persisted main-process queue and waits, so
UI and future MCP jobs share one line for the GPU. The renderer's own queue UI
is unchanged; moving it onto the main-process queue (so it can show MCP jobs)
is left for later. Jobs left queued/running at shutdown become `interrupted` on
next start and are not re-run.

Requirements:

- Submit returns a `job_id` immediately; work proceeds in the app.
- Jobs are persisted (survive the client disconnecting; ideally survive an app
  restart as "interrupted" rather than vanishing).
- States: queued → running (with phase/progress from `progressTracker`) →
  done / failed / cancelled. Failed jobs can be retried individually.
- One GPU job at a time by default; the queue serialises.
- Pushes state to the renderer so the UI shows queued/running jobs, whether
  they came from the UI or MCP.
- Jobs and Library items carry an optional **batch label** and a **source**
  (`ui` / `mcp`) so a music video's clips stay grouped and MCP output is never
  invisible.

## Tool surface (v1)

| Tool | Purpose |
|---|---|
| `list_capabilities` | Families, image-vs-video kind (`FAMILY_KIND`), curated fields and limits per family |
| `import_folder` | Import local images/audio from a path into the Library; returns IDs |
| `generate_image` | Submit an image job; returns `job_id`. Optional `style` (a saved style's name) is appended to the prompt |
| `list_styles` | The user's saved prompt styles (name and text), for `generate_image`'s `style` |
| `generate_video` | Submit an I2V job (source image by Library ID); returns `job_id` |
| `list_jobs` / `get_job` | Progress, results, errors; supports re-attaching later; filter by batch |
| `cancel_job` | Cancel queued/running job |
| `list_library` / `get_item` | Browse results; `get_item` includes a poster frame image for visual QA |
| `probe_media` | Duration, resolution, fps for a clip or audio file |
| `assemble_video` | Stitch clips and lay a backing track under them |

Design rules:

- Submit-and-poll, never a tool call that blocks for minutes. An optional
  "wait up to N seconds" parameter is fine.
- Only the curated field set per family is exposed (prompt, size/length, seed,
  source image), matching the existing philosophy. `list_capabilities` is the
  source of truth so the client doesn't guess.
- Inputs reference Library IDs. Paths are only accepted by `import_folder`,
  validated (exists, is a file/folder, allowed extensions).
- Errors are specific: "ComfyUI not reachable at <host>", not a timeout.
- Poster frames let the client look at each clip, notice a bad one and retry it.

## `assemble_video`

Inputs: ordered list of clips (Library IDs) with optional per-clip trim/duration,
one audio file, transition (`cut` or `crossfade` + duration), end behaviour,
output name.

- Wan 2.2 clips are silent, so the backing track is the only audio.
- **Length mismatch** (e.g. 15 × ~5 s vs. a 3:40 track) is the client's
  planning problem, solved with `probe_media` durations. The tool needs an
  explicit end behaviour: `trim_to_audio`, `trim_to_video` or `fade_out`, so it
  never silently produces a mismatched file.
- Hard cuts between uniform clips can use ffmpeg stream-copy (same template →
  same codec/resolution): fast and lossless. Any trim or crossfade re-encodes.
- Output is added to the Library as a normal item with the batch label.
- The existing `mp4Faststart.ts` post-processing should still apply to outputs.

### ffmpeg

Decided: look in three places, in order - a path chosen in Settings, the copy
bundled through the `ffmpeg-static` / `ffprobe-static` optional dependencies,
then PATH. Note those packages ship GPL-licensed binaries; the app is AGPL-3.0,
which is compatible, but check KVG_Standards' licensing guidance before a release
(see open items).

## Phasing

Status: 1-4 done. 5 not started.

1. **Queue + service layer.** Persisted job queue, extracted service layer,
   renderer uses it. Independently valuable (closes the TODO item).
2. **Local API + stdio shim + core tools.** `list_capabilities`,
   `generate_image`, `generate_video`, `list_jobs`/`get_job`/`cancel_job`,
   `list_library`/`get_item`. Settings toggle + token. Test against Claude
   Desktop.
3. **Batches and inputs.** `import_folder`, batch labels, poster frames.
4. **Stitching.** ffmpeg integration, `probe_media`, `assemble_video`.
5. **Later.** More templates as new primitives (upscale, img2img — see
   `TODO.md`), saved recipes as MCP prompts, Desktop extension packaging,
   possible headless mode.

## Worked example (target behaviour)

1. `import_folder("~/shoot")` → 15 Library IDs; `import_folder("~/song.wav")`.
2. `list_capabilities`, `probe_media(song)` → 3:40.
3. For each image: write a video prompt from the script; `generate_video(...)`
   with batch `"music-video-1"` → 15 job IDs.
4. Poll `list_jobs(batch=…)`; view poster frames; re-submit bad clips.
5. `assemble_video(clips=[…], audio=song, transition="crossfade", end="trim_to_audio")`.

## Security notes

- Localhost-only bind, bearer token, Origin rejection, off by default.
- Generation burns GPU and disk; the in-app toggle is the consent point.
- Path inputs validated; never serve or read arbitrary paths on request.
- Tool results and library contents are data; the API never executes anything
  derived from them.

## Open items

- **KVG_Standards check not done.** It was out of reach when this was written and
  built. Check it for an existing MCP/local-API convention and for ffmpeg
  bundling/licensing guidance, and reconcile this design with it.
- **Not verified in a packaged build or with real Claude Desktop.** Everything was
  exercised in dev Electron with a mock ComfyUI (see "Verification"). Still to try:
  the packaged app's asar/unpacked ffmpeg paths, and launching the shim from a real
  Claude Desktop config (`ELECTRON_RUN_AS_NODE`) on Windows/macOS.
- Imported and assembled items are visible to MCP clients (`list_library`) but not
  yet in the app's own Library page (see `TODO.md`).
- Job persistence across app restart: interrupted jobs are marked, not re-run.

## As built (phases 2-4)

**Files**

| File | Role |
|---|---|
| `src/shared/tools.ts` | Tool definitions (JSON Schema), used by both the API and the shim |
| `src/main/apiService.ts` | Runs the tools: validation, job submission, library, probing, assembly |
| `src/main/localApi.ts` | Loopback HTTP API: `GET /v1/health`, `GET /v1/tools`, `POST /v1/tools/<name>` |
| `src/main/library.ts` | `imports` table; presents generated (`gen-N`) and imported/assembled (`imp-N`) files as items |
| `src/main/mediaTools.ts` | ffmpeg discovery, probing, preview frames, and `planAssemble` (pure command builder) |
| `src/main/assembly.ts` | Background assembly runs (`assemblies` table) |
| `src/mcp/mcpProtocol.ts`, `src/mcp/shim.ts` | MCP stdio server; forwards to the API |

**How a client connects.** Settings -> "Other Apps (MCP)" turns the API on (off by
default) and shows a ready-to-paste `mcpServers` entry: it runs the app's own
binary with `ELECTRON_RUN_AS_NODE=1` on `dist/main/mcp/shim.js`, with
`KVGENIUS_API_FILE` pointing at `mcp-api.json` in the app's data folder. While the
API is on, the app writes its port and token there (mode 0600); the shim reads it
per call, so restarting the app needs no client reconfiguration. The token never
appears in the client config or the UI. Default port 47615, falling back to a
free one. Any other local client can use the HTTP API directly with the same file.

**Deviations from the plan**

- Assembly runs are their own background tasks (`get_assembly`), not rows in the
  job queue: they are CPU work and should not wait behind GPU jobs, and it keeps
  the queue generation-only. `cancel_job` also accepts `assembly_id`.
- Items are addressed as `gen-<id>` / `imp-<id>` strings rather than bare
  integers, because generated and imported files live in different tables.
- Imported files are referenced in place, never copied.
- `generate_video` takes `seconds` (snapped to quarter seconds) instead of frames.

**Security posture.** Loopback bind; bearer token (constant-time compare); any
request with an `Origin` header or a non-loopback `Host` is refused; 1 MB body
limit; off by default.

**Data scope.** A client sees only what it created: jobs submitted through the API,
and the library items they produced, plus its own imports and assemblies. Anything
made in the app itself, and anything from before jobs were recorded, is treated
exactly like an id that does not exist (`not_found`, same message) - in listings,
`get_item`, `probe_media`, `get_job`, `cancel_job`, and as a source for
`generate_video` / `assemble_video`. Batch cancellation only touches the client's
own jobs even when an app job shares the label. Pins, prompt tabs, timing
stats, settings and the database have no tool at all. To use an app-made image, point
`import_folder` at the file. Enforced in `library.ts` (`clientOnly`) and
`ApiService` (`item()` / `clientJob()`), tested in `apiService.test.ts`.

**Remaining exposure.** `import_folder` will read any image/audio/video file at a path
the client names (and `get_item` then previews it). That is what lets you say "use the
images in this folder", so it is not restricted yet; the toggle plus the token are the
consent point. Restricting it to folders approved in Settings is the obvious next step.

**Verification.** `npm test` (56 tests: queue, tool logic against a real ffmpeg,
assembly planning, API auth/errors, MCP protocol, and the compiled shim run as a
subprocess) plus a manual end-to-end run: dev Electron under Xvfb with a mock
ComfyUI, driven through the real shim over stdio - import, three video jobs run
serially, previews, crossfade stitch with a track held to the audio length,
cancel, the UI's own `generate` IPC sharing the queue, and the Settings toggle
closing and reopening the port.
