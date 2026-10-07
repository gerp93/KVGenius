# Model profiles and model management plan

Status: **all five stages are built and open as stacked draft PRs** (merge in this order, retargeting each to
`main` as the one before it merges): #90 setup guide -> #92 stage 1 -> #93 stage 2 -> #94 stage 3a -> #95 stage 3b ->
#96 stage 4 -> #97 stage 5. Nothing has been run against a real ComfyUI; each PR lists what is unverified.

Where the build differs from the plan below:

- Stage 3 was split into **3a** (profiles core) and **3b** (header check, local-file import, picture reader, test render).
- The video LoRAs are **required**, not optional: the Wan workflow contains both, so the manifest lists six files.
- Sizes and licences are **not** in the manifest yet (an unverified number is worse than none).
- The header comparison uses the user's own working file as the reference, as planned; its thresholds are a judgement
  and have not seen real model files. A real bug caught on the way: safetensors calls a tensor's byte range
  `data_offsets`, not `offsets`.
- The Hugging Face download links follow ComfyUI's `split_files/<folder>/<file>` layout but were **not opened** from the
  build environment; a moved file shows as a 404 in the terminal.
- The test render for video (9 frames, 256 px, Fast quality) and the PowerShell download script are **unrun**.
- When ComfyUI fails a run, its own reason is now reported (was "finished with no output") - for every generation.

## Goal

A user should be able to run KVGenius against a different model of a family we already
support (a fine-tune, a merge, a distilled or non-distilled sibling) without anyone
shipping new code, and should never have to open ComfyUI's own editor to do it.

Three things make that work:

1. KVGenius knows which model files it needs and can tell what is missing.
2. A **profile** records which files and sampler defaults to use within a family.
3. A user can bring a model file from anywhere (a download, a USB stick), have it
   **validated**, and have it **put in ComfyUI's models folder** by the app.

## Non-goals

- A model catalog, search, or recommendations beyond the files our templates need.
- New model *families* (a different architecture needs its own template and node map;
  that stays a code change, as in CLAUDE.md).
- Arbitrary ComfyUI workflow import.
- Downloading from arbitrary sites, pasted URLs, or stored site tokens (see "Deferred").
- Managing a ComfyUI that runs on another machine (we can detect what it has, but can't
  place files in it).

## Terms

- **Family** - the architecture, which fixes the node graph, loader slots and text encoder.
  Today: `z-image` (renamed from `z-image-turbo`, see stage 2) and `wan22-i2v`.
- **Variant** - a particular model within a family: the shipped Turbo, a non-distilled
  base, a fine-tune or a merge.
- **Profile** - KVGenius's saved settings for one variant: a name, a file per loader slot,
  and sampler defaults. The family decides the graph; the profile only changes values
  inside it.
- **Slot** - one loader input in a template (e.g. `unet_name`, `clip_name`, `vae_name`).

## Decisions made

| Question | Decision |
| --- | --- |
| Renamed family key | `z-image`. `wan22-i2v` stays (it already names family + task). |
| First release scope | Images **and** video (shipped as separate PRs, one release). |
| Profile validation | File-header check **plus** a test render. |
| Getting a model onto disk | Import from a **local file** the user already has. |
| Download helper | Only the files in our manifest (public Hugging Face files, no token). |
| Site tokens / pasted URLs | Deferred (see below). |
| Video profiles | Carry files and defaults only; the Fast/High switch is unchanged. |
| Migration backup | Created automatically; never deleted automatically. |

## Stages

Each stage is its own PR, in this order.

### 1. Manifest, detection page, models folder

- `src/shared/modelManifest.ts`: plain data listing every file a family or feature needs:
  file name, slot/folder (`diffusion_models`, `text_encoders`, `vae`, `loras`,
  `upscale_models`), approximate size, licence, source link, and which feature needs it.
  File names must match the templates in `src/main/templates/` (add a test that checks
  this, so the two cannot drift).
- Detection: ask ComfyUI what it can see via `/object_info` for the loader nodes
  (`UNETLoader`, `CLIPLoader`, `VAELoader`, `LoraLoaderModelOnly`, `UpscaleModelLoader`);
  `listUpscaleModels` in `comfyui.ts` is the existing pattern. Works for any ComfyUI
  address, no disk access needed.
- A Models page shows, per family and feature, present / missing, with the exact folder,
  size and source link for each missing file.
- The setup guide's tables read from the manifest instead of hard-coded rows, and its
  connect step reports "N files still missing".
- **Models folder setting:** guess from the launcher where possible (a portable install's
  run script sits beside `ComfyUI/models`; ComfyUI Desktop's launcher is the app, not the
  models folder, so there the user chooses it), with a Change button and a saved setting.
  The guess must be checked (folder exists, has the expected subfolders) before it is
  trusted. `extra_model_paths.yaml` redirects are invisible to us, hence the Change button.

### 2. Rename `z-image-turbo` to `z-image`

- Template file, node map and registry in `comfyui.ts`, `FAMILY_KIND`, `origin.ts`
  (two places), `apiService.ts` / `tools.ts` (list and default), the Generate default, docs
  and about 50 test occurrences.
- **Startup migration** in `db.ts`, beside `migrateSchema`: one transaction that updates
  `generations.model_family`, `jobs.family` (plus the family in saved job params, normalised
  on read too) and `timing_stats.family`. Idempotent. Before it first changes anything, copy the
  database file to a dated backup beside it. If it fails, roll back and keep running; the
  alias below keeps the app working on unmigrated rows.
- The old key stays accepted as an alias for MCP/API callers.
- Test: in-memory database with old-key rows in all three tables; run the migration;
  check the rename; run again and check nothing changes.
- Risk to note in release notes: an older version opening a migrated database will not
  recognise the key. Only matters for people sharing one database file across versions.
- The backup is **never deleted by the app**: the user removes it by hand when they are satisfied. Settings > Library & Data should say where it is so it is not forgotten.

### 3. Image profiles, validation and local-file import

**Storage.** A `model_profiles` table (id, family, name, files as JSON per slot, sampler
defaults, last test result, created/updated). The built-in profile is not a row; it is
generated from the manifest defaults so it cannot be deleted or drift.

**Generation.** The node map gains the loader slots and sampler fields
(`unet_name`, `clip_name`, `vae_name`, steps, CFG, sampler, scheduler, shift). The profile
is resolved *before* the job is queued, as styles are, so the job and the Library record
carry the full resolved values. The generation record stores the profile name (a label
only) plus the resolved file names; Re-rack uses the record, never the live profile, so
editing or deleting a profile never changes past generations. New columns are added the
way `migrateSchema` already does it. The UI's `generate` and the MCP/API `generate_image`
(`model` argument) both go through this; `list_models` lists profiles.

**Models page / editor** (same layout as Styles: list left, editor right):

- Pick a family first; it decides the slots and default fields.
- Each slot is a dropdown of files ComfyUI actually has, or **"From a file on this
  computer"** (below).
- Defaults start from the family's shipped values.
- **Read settings from an example image**: ComfyUI-saved PNGs embed the workflow, including
  the sampler's steps, CFG, sampler and scheduler; fill the fields from it when present.
  Many shared images have their metadata stripped, so this is best-effort.
- A "distilled / standard" question that sets starting values. Numbers for a non-distilled
  variant are **not** hard-coded until we have confirmed what the authors recommend; until
  then those fields start blank and flagged as a guess. A name containing turbo, lightning
  or lcm may pre-select "distilled", never set numbers silently.
- A link to the model's page next to the fields.
- Status per profile: tested and working, a file is missing from ComfyUI now, or untested.

**Validation** (both, as decided):

1. *File-header check* (safetensors only). Read the header (8-byte length + JSON) of the
   candidate and compare tensor names and shapes - not data types, so fp8 and bf16 copies
   of one model still match - against the header of the user's own working built-in file
   for that slot. This needs no shipped fingerprints (we cannot fetch the reference files
   from this environment) and works for any family. The header also gives each tensor's
   byte range, which implies the expected file size; a smaller real size means a truncated
   download, reported as such.
2. *Test render*: a tiny, few-step render with the profile. ComfyUI rejects mismatched
   weights; a pass shows the test picture so the user can judge the defaults. It works for a
   remote ComfyUI where the header check cannot (no local file).

GGUF and old `.ckpt` files cannot be header-checked and GGUF needs extra ComfyUI nodes;
say so in the UI rather than let them fail oddly. Warn on `.ckpt`/`.pt` (can contain code);
softer for `.pth` in `upscale_models`, which is the usual upscaler format.

**Local-file import.** Browse or drag-and-drop a file onto a slot, then:

1. Check format, size against the header, and architecture against the known-good file -
   **before** copying, so a wrong file never lands in ComfyUI's folder.
2. Show the result; on a pass copy (or move, as an option) into the right subfolder of the
   models folder, keeping the original file name (the name is the template's identity).
3. Ask before overwriting an existing file; check free disk space first; show progress;
   a cancel removes the half-written file.
4. Re-query ComfyUI and select the new file in the slot.

Only possible when ComfyUI is local and the models folder is known; otherwise the app
shows the exact folder to put the file in. Hard links are avoided at first (cross-drive and
permission limits). Open: whether ComfyUI's loader lists see a new file without a restart
(believed yes, not verified); if not, the step prompts a restart / `R`.

### 4. Video profiles

The same mechanism for `wan22-i2v`: slots for the high- and low-noise diffusion models,
the text encoder, the VAE and the two 4-step LoRAs. A profile carries those files and
defaults; the Fast/High switch (`129:131`, `shared/videoQuality.ts`) stays as is and still
selects between the LoRA path and the full path. Validation is the same two checks (a test
render of a very short, small clip). Video is much heavier, so the test render must be
clearly labelled as slow.

### 5. Manifest-only download helper

A "Download missing files" button on the Models page for the handful of manifest files
(public Hugging Face files, no account). It writes a small script and opens it in a terminal
window, so the terminal shows progress and a re-run resumes (`curl -L -C -`, PowerShell on
Windows).

- URLs and paths are passed as quoted arguments, never built into a command string; only
  `http`/`https`; show each URL and destination and ask for confirmation.
- Terminal launching differs per OS; on failure fall back to showing the commands to copy.
- Needs the models folder from stage 1. **Direct file URLs must be verified** (they were
  not reachable from the build environment) before this ships.

## Deferred

- **Site tokens** (Hugging Face gated files, Civitai) and pasted download URLs. With
  local-file import, users fetch other models in their browser and import the file, so
  neither is needed. If added later: store with Electron `safeStorage` (refuse to save when
  encryption is unavailable), pass via an environment variable not the command line, one
  token per host and sent only to it, show "set / unset" only. Check Civitai's current
  token mechanism first.
- **Tuning grid**: render one seed across several steps/CFG values and pick by eye
  (possibly reusing the Compare overlay). There is no reliable way to derive steps/CFG from
  a model file; the example-image reader and the grid are the practical sources.
- Recommended variants for a family: none shipped. If ever: must pass the same validation,
  come from the author or a well-known repository, and have a licence allowing normal use.

## Checks before building

- Licence of each manifest file (record it in the manifest, show it in the UI).
- Direct download URLs for stage 5.
- What the authors recommend for a non-distilled Z-Image variant, if we want to ship numbers.
- Whether the estimator (`timingStats.ts`, grouped by family) needs the profile too, since
  differently sized models in one family could blur estimates.
- Whether new files appear in ComfyUI's loader lists without a restart.
