# MCP integration plan

Status: **plan only, nothing built.** Written from a design conversation; revisit
before starting work.

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

Today `generate()` is one blocking IPC call and `comfyui.ts` tracks a single
`currentPromptId` / abort controller (this is the "Job queue" item in
`TODO.md`). MCP needs this to exist first, because video jobs take minutes and a
batch may be 15 of them.

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
| `generate_image` | Submit an image job; returns `job_id` |
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

No ffmpeg usage exists in the repo today. Decide before implementing: bundle a
binary vs. detect a system install vs. a Settings path override. Bundling is the
friendlier default but builds can carry GPL obligations depending on
configuration, so check the licensing/packaging convention in KVG_Standards
first (see open items).

## Phasing

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

- **KVG_Standards check not done.** It was out of reach when this was written.
  Before building, check it for an existing MCP/local-API convention and for
  ffmpeg bundling/licensing guidance, and reconcile this plan with it.
- ffmpeg: bundle vs. detect vs. override (above).
- Job persistence across app restart: how much to promise in v1.
- Whether the shim is a flag on the main binary or a separate entry point
  (affects packaging).
- Default queue concurrency and whether ComfyUI's own queue should be used
  instead of ours.
