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
  before this have none and need a source chosen again. **The copy is made when the job is queued** (`keepJobSources`, via `JobQueue`'s `prepare` hook
  in `main.ts`), and the job is pointed at the copy - so a job waiting in the queue never depends on where the original was, and an original that is
  already gone is refused at once with a clear message instead of failing later. A job that ends without a result (failed or cancelled) lets go of
  its copies (`onUnfinished`), and `releaseSourceImage` never deletes a file a waiting or running job still needs.
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
- **Styles come in two kinds - Style and Element - both just text** (`shared/styles.ts`, `main/styles.ts`, `pages/Styles.tsx`). A
  *style* is a general look ("1930s movie poster, bold lithograph...") and a picture uses at most one; an *element* is a reusable part
  of the picture (an outfit, a character) and a picture can use any number. Both live in the `styles` table (`kind` column, default
  `'style'`, added to older databases by `migrateSchema`; names unique ignoring case across both kinds). Image mode only. Generate has a
  Style dropdown (kind style only) and an Elements chip list with an "add" dropdown (per prompt tab: `PromptSlotData.styleId`,
  `elementIds`; default none). `combinePrompt` appends at submit time, in this order: the prompt, the elements (in the order picked),
  the style last (`extraWording`) - the UI's `generate` and the MCP/API `generate_image` (`style` by name, `elements` as a list of names;
  `list_styles` lists both with their `kind`; a name of the wrong kind is refused with a pointer to the right argument) both do this
  *before* the job is queued. So the job, the Library record, Re-rack, pins, hidden words and group-by-prompt all see the one full prompt
  that was sent, and with nothing picked `combinePrompt` returns the prompt untouched. `generations.style_name` is only a display label
  (`styleLabel`: "Anime + Red coat"); the text is never re-derived from it. Re-rack / a prompt from Library > Prompts load that full
  prompt with no style or elements picked (else they would be added twice). Editing or deleting one never touches past generations.
- **The Library's card size is one slider** (`CardSizeSlider.tsx`, `hooks/useCardScale.ts`) in the toolbar of Output, Prompts and the Trash. It is a
  multiplier (0.75-2, default 1, remembered in `localStorage`) on each page's own target row height that `justifyRows` packs to, so bigger cards mean
  fewer to a row and each page keeps its own default size. A card's action buttons wrap onto a second line when the card is narrow
  (`.library-card__actions`), so a small size does not crush them; below about 0.75 a tall, thin card's overlay buttons start to collide.
- **The Library list stays where it is when a card is deleted or the details panel opens/closes** (`hooks/useScrollKeeper.ts`, used by Output and
  Prompts; cards carry `data-card-id`). Deleting from the panel closes it, the grid widens and gets shorter, and a list scrolled far down kept its
  numeric scroll position - landing somewhere much deeper (it looked like a jump to the bottom). `hold()` notes which card is at the top of the view
  and where just before such a change, and every render for the next ~0.7 s puts that card back at the same place.
- **Anything showing "ComfyUI is not reachable" fills in by itself once ComfyUI is started** (`hooks/useRetryWhenReachable.ts`: polls every 3 s while
  waiting and retries) - the upscale controls in the details panel and Tools > Upscale. The "try again" link stays as the manual way.
- **Group by prompt is an opt-in filter, not the default view** (`listPromptStacks` in
  `db.ts`). A stack card must take exactly the width `justifyRows` gave it (its stacked edges are box-shadow, no margin): a
  row even a few pixels too wide made the page grow, which fitted more cards per row, until everything sat in one row. Only *exactly* equal prompts stack. The cover is pinned > favorite > newest;
  stacks are ordered and cursor-paged by their newest item (`groupNewestId`), and the
  Library's filters apply to items before they are grouped. Images and videos stack
  separately; trashed items are never in a stack.
- **Navigation is a left sidebar, not a top bar** (`components/SideNav.tsx`/`.css`, rendered by `App.tsx`). Pages are
  grouped in sections (Create: Image / Video, Upscale, Styles; Library: Output/Prompts/Sources/Trash; Utilities:
  Timing/Hardpoint) in the `SECTIONS` data at the top of `SideNav.tsx` - add a page there. The footer holds
  Settings, Setup guide, the app-wide Show hidden switch, the ComfyUI status (click to launch when unreachable) and the
  graphics card ComfyUI runs on (`getGpuInfo` -> ComfyUI's `/system_stats`, `shared/gpuInfo.ts`; refreshed with the
  connection check, so it is also right for a ComfyUI on another machine). **Models is a tab of Settings**
  (`/settings?tab=models`; the old `/models` redirects there), reached from Generate's "Model settings" link beside the
  Model dropdown, Settings > ComfyUI and the setup guide - not from the rail.
  **The Generate page's prompt tabs are listed in the sidebar under Image / Video** (`PromptTabs` in `SideNav.tsx`):
  Generate still owns them (`usePromptSlots`, each tab a full copy of its form) and publishes `PromptTabsModel`
  (`utils/promptTabs.ts`) through `onTabsChange`; the sidebar only lists them (click selects and goes to Generate,
  double-click renames, ✕ closes, ＋ New tab). Only that list scrolls when the window is short - the rest of the rail keeps
  its place - down to a floor of three tabs, below which the whole rail scrolls as a last resort. Thin shows each tab as
  its IMG/VID tag. Footer groups are separated by rules.
  Every flex ancestor between the shell and a page needs `min-width: 0` (`.app-body`, `.app-main`, `.app-content`): otherwise a
  page wider than the window widens the whole body and pushes the details panel (`.details-slot`) off the right edge.
  Pages use the whole width (no max-width columns): Settings tabs flow their cards into as many 520px+ columns as fit
  (`.settings-panel--cards`: columns 520-640px wide packed from the left and capped at three, so a huge screen leaves plain space on the right instead of stretched cards; the Models tab is one wide editor), and Styles, Upscale and the setup guide fill the page.
  The rail is full (labels; a section's head folds it) or thin (**icons only, no hover flyouts**: each page keeps its own
  icon, each section shrinks to a caption, so every page is one click away). Thin/full and folded sections are remembered
  in `localStorage`. Every page needs an icon and a `title` for that reason. The shell is a row: sidebar, then
  `.app-body` (page + queue bar, then the details slot).
- **The queue is a bar along the bottom of the app shell, on every page** (`App.tsx`,
  `QueuePanel.tsx`, `.queue-bar`). Panels are docked flush - no margin or padding around them, only
  a divider line on the inner side. Folded (the default) it is a slim status strip: count, a tile
  per running/waiting/failed job, what the running job is doing and a progress line. Open, it is a
  drawer (`clamp(200px, 30vh, 320px)`) of horizontal cards that *pushes the page up* (it is in the
  flex column, not an overlay). Clicking anywhere on its head bar opens or folds it (not only the ▲/✕ button). It always starts folded when the app opens (open/folded is not remembered between launches). The Library's details
  panel is not collapsible: it docks to the right edge only once an item is clicked and ✕ closes it
  (its width scales with the screen) - which is why `.library-page` has no padding of its own and
  `.library-output__main` carries it. Likewise Generate has no page padding: the prompt-tab rail
  (`.prompt-slots-rail`, tabs share its height up to a cap and shrink as more open) and the preview
  run edge to edge and the form carries the padding. A card for a job made from a picture (a video, image to image, inpainting, outpainting, a picture upscale) shows that reference picture small in its bottom-right corner (`ReferenceThumb`); a video upscale shows the first frame of its source video. A finished card in the queue can be favorited,
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
  Models tab) and a "Setup guide" item in the sidebar's footer (a red dot while ComfyUI cannot be reached) - not
  an app-menu item, which KVG_Standards keeps to View/Help basics. External links go through the `openExternal` IPC (http/https only,
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
- **The Models tab is one list of every model, each with its own files** (`components/ModelProfiles.tsx`, grouped Image / Video /
  Upscaling). The shipped starter models (`PROFILE_FAMILIES`' `builtInName`: Z Image Turbo, Wan 2.2) are ordinary entries - read-only,
  showing their manifest files table, status and Download button - next to the user's own saved models (editable, with a
  per-slot file picker and import). Each entry shows a readiness word (`readinessLabel`, `summarizeSlots` in `shared/modelStatus.ts`).
  **+ New is a wizard** (type: Image / Video / Upscaling -> family, only when the type has several -> name and files -> settings,
  image families only -> test and save; Upscaling is just type -> the file). Steps are computed in `ModelProfiles.tsx`
  (`stepIds`), so a new family under a type needs only an entry in `PROFILE_FAMILIES`. A saved model is edited on one page.
  Upscale models are chosen by the user per run, so that entry lists what ComfyUI has and offers the same file import as any model
  slot (`UPSCALE_IMPORT_FAMILY` / `importSlot` in `shared/modelFamilies.ts` - a pseudo family with one slot, not a profile kind). Do not describe any model as "the" built-in one
  in the UI - it is only the first one the app shipped with.
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
- **The model editor's file dropdowns only offer files that fit together, worked out from the files' own headers** (`shared/modelTraits.ts`,
  `main/modelTraitsReader.ts`, the `getModelFileTraits` IPC, `ModelProfiles.tsx`). `readTraits` reads, from tensor names and shapes, what a file
  is (`arch`) and the one number it must agree with its neighbours on: a diffusion model's text-embedding width and latent channels, an encoder's
  output width, a VAE's latent channels (and picture vs video VAE; Wan t2v vs i2v by patch-embedding input channels). `checkFit(family, slot,
  candidate, chosen)` compares a candidate with the family's needs and with what the *other* slots picked, so the template's default files are
  not the filter - a pick in one slot narrows the others, in any order. Files that conflict are hidden behind "Show every file" (shown with ⚠ and a
  reason); a file that cannot be read or is not recognised is `unknown` and never hidden; LoRA slots are not checked. Only headers are read, cached
  by path/size/mtime, and only inside the models folder. **The tensor names in `readTraits` were written from memory of the model code, not checked
  against real files - verify them against real headers (Z-Image, Qwen3-4B, Flux VAE, Wan 2.2 t2v/i2v, UMT5, Wan VAE) when Hugging Face is
  reachable.** A wrong name makes a file read as `unknown` (nothing filtered), a wrong shape position could hide valid files.
- **The download helper only fetches manifest files, by running a script in a terminal** (`downloadScript.ts`,
  `downloadLauncher.ts`, `ModelDownload.tsx`). Links come from `ManifestFile.url` (Hugging Face, public, no token);
  anything else the user fetches themselves and imports. The user confirms a list of exactly what, from where, into
  which folder, then the script is saved to `userData/downloads/` and opened in a terminal (`cmd start powershell` on
  Windows, `open -a Terminal` on macOS, the first terminal found on Linux; if none opens the script is shown to copy).
  Every link and path is a quoted literal passed as an argument, never spliced into a command (`shellQuote`,
  `powershellQuote`; the POSIX script is run against a hostile path in a test). Files go to `<name>.part` and are
  renamed when whole, so running the script again resumes. Only missing files are fetched; a file in a subfolder is
  reported, not fetched again. A link that has moved shows as a 404 in the terminal and the rest carry on.
- **Image to image is Generate's image mode with a source image, and its own family** (`z-image-i2i`,
  `shared/imageToImage.ts`, `templates/z-image-i2i.json`, `main/imageToImagePatch.ts`). It is the text-to-image graph
  with the empty latent replaced by load picture -> scale to the output size (centre-cropped) -> VAE encode, and the
  sampler's `denoise` ("How much to change it", 0.05-1, default 0.6) says how much is re-drawn. Image mode has a Text → Image / Image → Image radio (the tab's `imageFromPicture`; the start
  block shows only for Image → Image, which cannot run until a picture is set). The picture comes from a file, a drop, or
  the Library (`LibraryPicker.tsx`, also offered for a video's source image); choosing one sizes the output to its shape
  and `imageFamilyFor` picks the family. It is held in the tab's `imageSourcePath`, apart from a video's `sourceImagePath`. It follows the supplied-picture pattern above - it is in
  `SOURCE_IMAGE_FAMILIES`, so the result keeps a copy, shows "Original", appears in Library > Sources and cannot be
  re-racked once the copy is gone - and has its own origin tag (Image -> Image). The loader/prompt/sampler/shift node ids
  are the same as text to image's, so **model profiles apply unchanged** (`profileFamilyKey`; `modelPatch.test.ts`
  holds both templates to Z-Image's slots). `generations.denoise` stores the strength; the duplicate guard compares it
  and the source image. MCP: `generate_image` with `source` (a library picture) and `strength`; naming the family
  `z-image-i2i` directly is refused. The details panel's "🎨 Image" and Sources' "Image to image" send a picture to
  Generate through the same request as "🎬 Video" (`VideoSourceRequest.target`).
- **Inpainting is image to image with an optional painted mask, and its own family** (`z-image-inpaint`, `INPAINT_FAMILY` in
  `shared/imageToImage.ts`, `templates/z-image-inpaint.json`, `fillInpaint` in `main/imageToImagePatch.ts`, `MaskEditor.tsx`).
  Under the source image on Generate, "Paint a mask..." opens a full-screen editor (brush, eraser, size, undo, clear, invert); the
  result is a black and white PNG (white = re-draw) that `saveMaskImage` stores in the sources folder (`main/maskStore.ts`, hash-named,
  PNG-checked). With a mask, `imageFamilyFor` picks the inpainting family; a different source image clears the mask. The workflow
  is the i2i graph plus: mask -> scaled the same way as the picture -> softened (ImageBlur) -> `SetLatentNoiseMask` on the encoded
  latent, and the decoded result is pasted back over the original with the same mask (`ImageCompositeMasked`), so everything
  outside the mask is the original pixel for pixel. Core nodes only. The mask is the second kept picture (`generations.mask_image_path`):
  kept like a source image (same folder, released with the last result using it as source *or* mask - `releaseSourceImage`, `emptyRows`),
  shown under "Original" in the details panel, needed for Re-rack (`keptFilesOf` / `useSourceMissing` check both), part of the duplicate
  guard, and `generate_image` over MCP refuses the family (a mask can only be painted in the app). Z-Image Turbo is not an inpainting
  model: how cleanly it handles a partial mask is unverified.
- **Text to video is Generate's video mode without a source image, and its own family** (`wan22-t2v`, `shared/textToVideo.ts`,
  `templates/wan22-t2v.json`). Video mode has a Text → Video / Image → Video radio (the tab's `videoFromPicture`, default image to video, as
  every config saved before it was); the source image field shows only for Image → Video, which cannot run until a picture is set, and
  `videoFamilyFor` picks the family. The template is the image-to-video graph with the load-picture node removed and `WanImageToVideo`
  replaced by an empty video latent (`EmptyHunyuanLatentVideo`), **keeping every node id**, so `WAN22_I2V_NODE_MAP` patches both and the
  Fast / High switch, length and size work unchanged. It uses Wan's separate text-to-video models and LoRAs (own `WAN_T2V_FAMILY` profile
  family, `SLOT_NODES['wan22-t2v']`, manifest feature `wan22-t2v`; the text encoder and VAE are shared files, which is why the manifest test
  only forbids repeats *within* a feature). It makes no use of a picture, so it is **not** in `SOURCE_IMAGE_FAMILIES`. Its origin tag is
  Text → Video. MCP: `generate_video` without `source` is text to video (with `source`, image to video; naming the wrong family for the
  arguments is refused). The T2V file names are unverified against the Hugging Face repo.
  Whenever a source image is set for image to video - chosen from a file, dropped, or picked from the Library - the video size follows its shape (`setVideoSizeToPicture`: long side 640, sides in 16s) so a portrait picture is not cropped into the square default.
- **A video can be extended** (`➕ Extend this video` in the details panel, `main/extendVideo.ts`, `planLastFrame` / `planJoin` in `mediaTools.ts`). The
  button reads the video's last frame with ffmpeg (`prepareVideoExtension`), keeps it as a source picture and opens Generate in Image → Video with
  that frame, the video's size and prompt (`VideoSourceRequest.extend`, the tab's `extendFromId`, shown as a note with "Make a separate clip
  instead"); the job carries `extendVideoId`. When the clip finishes, `generationService` joins it onto the end of the earlier video (re-encoded to the
  earlier video's size and frame rate, the clip's repeated first frame dropped) and the one longer video is what is kept as the new record; the
  original is untouched. `generations.extended_frames` holds how many frames came from before the clip, so the details panel shows the whole length
  while `length` stays the clip's (Re-rack re-makes the clip, not the join). If the join cannot be done (ffmpeg missing, the earlier video gone) the
  clip is kept on its own rather than losing the render. Only `wan22-*` videos can be extended. In `planJoin` the frame rate must come *last* in each
  chain - earlier, ffmpeg 7's concat repeats frames (a test with the bundled ffmpeg checks the frame count). Not available over MCP.
- **Outpainting is image to image with the picture extended beyond its frame, and its own family** (`z-image-outpaint`, `OUTPAINT_FAMILY` and
  the padding helpers in `shared/imageToImage.ts`, `templates/z-image-outpaint.json`, `fillOutpaint` in `main/imageToImagePatch.ts`,
  `ExtendField.tsx`). Under the source image on Generate, "Extend beyond the frame" sets pixels to add per side (left/top/right/bottom,
  `GenerationParams.outpaint`, stored on `generations.outpaint` as `"l,t,r,b"`); any side above 0 makes the run outpainting
  (`imageFamilyFor`'s fourth argument, which wins over a mask - setting one clears the other). The workflow is the inpainting graph with the
  painted-mask loader replaced by `ImagePadForOutpaint` (grey padding plus a mask of exactly the new area, feathered 40 px into the original),
  both scaled to the output size **without cropping** and the result pasted back over the padded original, so the original is untouched.
  The output size is the *whole extended canvas* (`outpaintOutputSize`: long side kept within 1024-1536, sides in 64s), set by Generate from the
  source's size - the Size menu is replaced by a note. **The new area does not start from grey**: it starts from a heavily blurred stretch of the whole
  picture composited under the original (`op-bg` / `op-bgblur` / `op-init` in the template) and is only partly re-drawn - `denoise`, "How much to
  invent in the new area", default 0.8 (`OUTPAINT_DEFAULT_DENOISE`), on the same slider as image to image. At 1 the model ignores that start and
  draws the area from the prompt alone, which produced an unrelated picture in the margin (the first version did exactly that); a prompt that
  describes the whole scene has the same effect, so the form says to describe what continues beyond the edges (the noise mask keeps the original
  in place while the new area is drawn). It follows the supplied-picture pattern (`SOURCE_IMAGE_FAMILIES`: kept copy, "Original", Library > Sources, Re-rack disabled once
  the copy is gone; the details panel says how far it was extended), has its own origin tag (Outpainted), uses Z-Image's profiles, and the duplicate
  guard compares the padding and strength. MCP: `generate_image` with `source`, `extend` ({left,top,right,bottom}) and optionally `strength`; naming the family directly is refused.
  Unverified against a real ComfyUI: that the graph is accepted and how well Z-Image Turbo continues a picture at the edge.
- **`shared/modelManifest.ts` lists every model file the templates ask for** (name, ComfyUI folder, role, source) and
  `main/modelManifest.test.ts` pins it to the template JSON - change a template's loader file and that test fails
  until the manifest matches. Which files exist comes from ComfyUI's own loader lists (`/object_info`,
  `listInstalledModels`), falling back to a scan of the models folder when ComfyUI is down (`modelsFolder.ts`;
  the folder is the user's choice, else guessed from the launcher and only trusted if it looks like a `models`
  folder; it is set, opened (`openModelsDir`) and warned about in Settings > ComfyUI, and the Models tab only shows a
  warning when it is not usable). A file inside a subfolder is reported as such, never as installed: the template asks for the plain name.
  The Models tab (Settings) and the setup guide both render from this.
- **Compare picks a best, "winner stays"** (`shared/tournament.ts`, `CompareOverlay.tsx`):
  A or B, the pick meets the next item, N-1 questions; "Neither" drops both; undo is a
  history of states. Afterwards the rest can go to the Trash - never favorites or pinned
  (it calls `trashGenerations` without `includeKept`).
