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
  family (`z-image.json` for image mode, `wan22-i2v.json` for video
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
  a video in place. The copy is deleted with the last result that uses it; videos made
  before this have none and need a source chosen again.
  **This is the pattern for anything made from a supplied picture**, not a video-only rule:
  `shared/sourceFamilies.ts` lists the families (`wan22-i2v`, `upscale-image`), and a new
  tool that works on a picture goes in that list. For every family in it: the generation
  keeps a copy (`generationService.ts`); the details panel shows it as "Original"; Library >
  Sources lists the kept pictures (Upscale / Make video from them); and Re-rack is disabled,
  with "Not available - the source image is missing.", once the copy is gone from disk
  (`useMissingSources.ts`, checked by the `sourceImagesMissing` IPC). An upscale's Re-rack
  opens Tools > Upscale with its original instead of loading Generate. A record with no
  recorded source (made before copies were kept) is left re-rackable as it always was.
- **Deleting is two steps, and there is no delete confirmation** (`trash.ts`,
  `cleanupScheduler.ts`, `shared/cleanup.ts`). Step 1: the Delete button anywhere (and the
  cleanup, for unfavorited/unpinned items older than N days) only moves the item to
  `output/trash` (record kept, `trashed_at` set), so it can be undone or restored. Step 2:
  emptying the Trash sends the files to the OS Recycle Bin (`shell.trashItem`, injected as
  `Recycle`), and an item the bin refuses stays in the Trash. There is no hard delete.
  The automatic cleanup never trashes a favorite or pinned item (`moveToTrash` refuses
  unless `includeKept`, which only an explicit per-item Delete sets). Every Library
  listing, and the MCP `library` views, filter `trashed_at IS NULL` (`filterSql`). The two
  automatic steps - move to Trash, empty Trash - are separate options, **both off by
  default** (Settings > Library Cleanup); turning one on starts its clock, so its first run
  is a day later.
- **Styles are optional, one per generation, image mode only, and just text** (`shared/styles.ts`,
  `main/styles.ts`, `pages/Styles.tsx`). A style is a named snippet of wording ("1930s movie poster,
  bold lithograph...") stored in the `styles` table (names unique ignoring case). Generate's Style
  dropdown (per prompt tab, `PromptSlotData.styleId`, default none) shows the style read-only under the
  prompt and `combinePrompt` appends it at submit time - the UI's `generate` and the MCP/API
  `generate_image` `style` argument (by name; `list_styles` lists them) both do this *before* the job is
  queued. So the job, the Library record, Re-rack, pins, hidden words and group-by-prompt all see the one
  full prompt that was sent, and with no style `combinePrompt` returns the prompt untouched - nothing
  changes from before styles existed. `generations.style_name` is only a label for the details panel; the
  text is never re-derived from it. Re-rack / a prompt from Library > Prompts load that full prompt with no
  style picked (else the style would be added twice). Editing or deleting a style never touches past
  generations.
- **Group by prompt is an opt-in filter, not the default view** (`listPromptStacks` in
  `db.ts`). Only *exactly* equal prompts stack. The cover is pinned > favorite > newest;
  stacks are ordered and cursor-paged by their newest item (`groupNewestId`), and the
  Library's filters apply to items before they are grouped. Images and videos stack
  separately; trashed items are never in a stack.
- **The queue is a bar along the bottom of the app shell, on every page** (`App.tsx`,
  `QueuePanel.tsx`, `.queue-bar`). Panels are docked flush - no margin or padding around them, only
  a divider line on the inner side. Folded (the default) it is a slim status strip: count, a tile
  per running/waiting/failed job, what the running job is doing and a progress line. Open, it is a
  drawer (`clamp(200px, 30vh, 320px)`) of horizontal cards that *pushes the page up* (it is in the
  flex column, not an overlay). Open/folded is remembered in `localStorage`. The Library's details
  panel is not collapsible: it docks to the right edge only once an item is clicked and ✕ closes it
  (its width scales with the screen) - which is why `.library-page` has no padding of its own and
  `.library-output__main` carries it. Likewise Generate has no page padding: the prompt-tab rail
  (`.prompt-slots-rail`, tabs share its height up to a cap and shrink as more open) and the preview
  run edge to edge and the form carries the padding. A finished card in the queue can be favorited,
  pinned or deleted (to the Trash, no confirmation); those handlers live in `App.tsx` and announce
  the change with `utils/generationChanges.ts` so a mounted Library list or Generate form follows
  (a favorite also moves the file, so the new path comes along). Clicking a finished card opens the
  same `LibraryDetails` panel, app-wide (`App.tsx`, looked up from the queue by id); it and a Library
  page's own panel close each other so only one shows. Both render through `DetailsDock` (a portal into
  `.details-slot`), a full-height column at the right of the window, so the queue bar only spans the
  page to its left.
  The panel's width scales with the screen (`.library-panel`); once it is big enough that the picture
  would come out clearly larger, it snaps to a two-column layout - the picture in a full-height column,
  the details beside it - decided by `shared/detailsLayout.ts` from the panel's measured size and the
  picture's aspect ratio (tall pictures gain a lot, wide ones usually stay stacked).
- **Every Library item carries an origin tag** (`shared/origin.ts` -> `OriginBadge`): Text → Image,
  Image → Video, Upscaled or GIF, from its `modelFamily`; an unknown family gets no tag rather than a
  wrong one. It sits in the badge row of an Output card, a Prompts tile and the details panel header -
  add a family to `generationOrigin` when adding a model.
  Library > Output can show only chosen origins (chips at the left of its toolbar; none on = all). Each
  tab (Images / Videos) offers only the origins that can occur on it (`originsForKind`) and keeps its
  own selection, and the tab counts follow each tab's own filter (`originsByKind`):
  `LibraryListOptions.origins` -> `originCondition` in `db.ts` (a fixed family table, never caller text),
  applied to listings, stacks, counts and select-all refs; `passesOriginFilter` keeps freshly made items
  (finished upscales, GIFs) consistent with it.
- **The ComfyUI setup guide is a data-driven stepper** (`pages/Setup.tsx` on `/setup`, `components/Stepper.tsx`).
  Steps are plain `Step[]` data; the file names/folders in its tables must match the templates in
  `src/main/templates/` (change one, change the other). It polls `checkComfyUIConnection` so the connect step
  turns green by itself. Entry points are two buttons at the top of Settings > ComfyUI (the guide, and the
  Models page) and a "?" icon at the right end of the top bar (a red dot while ComfyUI cannot be reached) - an
  icon rather than a labelled link because the bar is already full, and not an app-menu item, which KVG_Standards
  keeps to View/Help basics. External links go through the `openExternal` IPC (http/https only,
  `shared/externalUrl.ts`), never a plain `<a href>`.
- **The image family key is `z-image`** (the family), not `z-image-turbo` (one variant of it). The old key is
  retired but still accepted everywhere a family comes in (`shared/families.ts`: `canonicalFamily`, used by
  the MCP/API, the job store, the template lookup and origin tags), and `familyMigration.ts` rewrites it in
  `generations`, `jobs` and `timing_stats` at startup, in one transaction, after copying the database to
  `<db>.pre-family-rename` (`VACUUM INTO`). That copy is never deleted by the app; Settings > Library & Data
  lists it so the user can. Add a future rename to `LEGACY_FAMILY_KEYS` and the migration picks it up.
- **Model profiles are values inside a family's graph, never a different graph** (`shared/modelFamilies.ts`,
  `shared/modelProfiles.ts`, `main/modelProfiles.ts`, `main/modelPatch.ts`, Models page). A profile is a name, a
  file per loader slot and sampler values; the built-in model is not a row, it is the shipped template. Like
  styles, a profile is resolved *before* the job is queued (UI `generate` and MCP `generate_image`'s `model`
  argument): `GenerationParams.modelName` / `modelSettings` carry the exact files and sampler, and
  `generations.model_name` / `model_settings` keep them, so the queue, Re-rack and the details panel never depend
  on the profile still existing. With no profile `modelSettings` is absent and the template runs exactly as
  shipped. Re-rack picks a saved model again only if it still means *exactly* the recorded settings
  (`profileMatchesSettings`), otherwise it falls back to the built-in one and says so. The duplicate guard compares
  the model too. `modelPatch.ts`'s node ids must match the template (`modelPatch.test.ts` checks it); a new
  family with profiles needs an entry in `PROFILE_FAMILIES` and `SLOT_NODES` (plus `SAMPLER_NODES` if the app drives
  its sampler). A family with `sampler: null` - Wan video - has **files only**: its quality stays the Fast / High
  switch (`videoQuality.ts`), a video profile's sampler columns hold the unused `NO_SAMPLER` placeholder, its
  `ModelSettings` carry no sampler fields, and `profileSettings` needs the family to know which. All six Wan files
  (two models, text encoder, VAE, two LoRAs) are slots because the workflow contains both LoRAs. A video model's
  test render animates a generated solid-colour picture (`solidPng.ts`), 9 frames at 256 px.
- **A model file is looked over before it is used or copied** (`main/safetensors.ts`, `main/modelImport.ts`).
  Only the header of a `.safetensors` file is read: tensor names and shapes (never data types, so fp8 and bf16
  copies of one model match) are compared with the *known-good file for that slot* - the one the shipped
  template loads, found in the models folder - so no fingerprints ship with the app and a new family needs
  none. A header also implies the exact file size, so a cut-off download is caught. `.gguf` is refused,
  `.ckpt`/`.pt` warned about (they can run code; `.pth` in `upscale_models` is the normal upscaler format and is
  not). Import copies to `<name>.part` and renames only when whole, keeps the original name (the template
  finds a file by name), never overwrites without asking, and only files the user chose or dropped are
  accepted (`pickedModelFiles` in `main.ts`). The test render goes through `JobQueue.runExclusive`, which
  refuses while a job runs and holds the queue while it works, so it never shares the GPU or the module-level
  in-flight prompt with a queued job. "Read settings from a picture" parses PNG text chunks only
  (`imageMetadata.ts`); pictures with the metadata stripped yield nothing, and nothing is guessed.
- **The download helper only fetches manifest files, by running a script in a terminal** (`downloadScript.ts`,
  `downloadLauncher.ts`, `ModelDownload.tsx`). Links come from `ManifestFile.url` (Hugging Face, public, no token);
  anything else the user fetches themselves and imports. The user confirms a list of exactly what, from where, into
  which folder, then the script is saved to `userData/downloads/` and opened in a terminal (`cmd start powershell` on
  Windows, `open -a Terminal` on macOS, the first terminal found on Linux; if none opens the script is shown to copy).
  Every link and path is a quoted literal passed as an argument, never spliced into a command (`shellQuote`,
  `powershellQuote`; the POSIX script is run against a hostile path in a test). Files go to `<name>.part` and are
  renamed when whole, so running the script again resumes. Only missing files are fetched; a file in a subfolder is
  reported, not fetched again. A link that has moved shows as a 404 in the terminal and the rest carry on.
- **Image to image is Generate's image mode with a start picture, and its own family** (`z-image-i2i`,
  `shared/imageToImage.ts`, `templates/z-image-i2i.json`, `main/imageToImagePatch.ts`). It is the text-to-image graph
  with the empty latent replaced by load picture -> scale to the output size (centre-cropped) -> VAE encode, and the
  sampler's `denoise` ("How much to change it", 0.05-1, default 0.6) says how much is re-drawn. Choosing or dropping a
  picture switches the family (`imageFamilyFor`) and sizes the output to its shape; the picture is held in the tab's
  `imageSourcePath`, apart from a video's `sourceImagePath`. It follows the supplied-picture pattern above - it is in
  `SOURCE_IMAGE_FAMILIES`, so the result keeps a copy, shows "Original", appears in Library > Sources and cannot be
  re-racked once the copy is gone - and has its own origin tag (Image -> Image). The loader/prompt/sampler/shift node ids
  are the same as text to image's, so **model profiles apply unchanged** (`profileFamilyKey`; `modelPatch.test.ts`
  holds both templates to Z-Image's slots). `generations.denoise` stores the strength; the duplicate guard compares it
  and the start picture. MCP: `generate_image` with `source` (a library picture) and `strength`; naming the family
  `z-image-i2i` directly is refused. The details panel's "🎨 Image" and Sources' "Image to image" send a picture to
  Generate through the same request as "🎬 Video" (`VideoSourceRequest.target`).
- **`shared/modelManifest.ts` lists every model file the templates ask for** (name, ComfyUI folder, role, source) and
  `main/modelManifest.test.ts` pins it to the template JSON - change a template's loader file and that test fails
  until the manifest matches. Which files exist comes from ComfyUI's own loader lists (`/object_info`,
  `listInstalledModels`), falling back to a scan of the models folder when ComfyUI is down (`modelsFolder.ts`;
  the folder is the user's choice, else guessed from the launcher and only trusted if it looks like a `models`
  folder). A file inside a subfolder is reported as such, never as installed: the template asks for the plain name.
  The Models page (`/models`) and the setup guide both render from this.
- **Compare picks a best, "winner stays"** (`shared/tournament.ts`, `CompareOverlay.tsx`):
  A or B, the pick meets the next item, N-1 questions; "Neither" drops both; undo is a
  history of states. Afterwards the rest can go to the Trash - never favorites or pinned
  (it calls `trashGenerations` without `includeKept`).
