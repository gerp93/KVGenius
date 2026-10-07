import type { CleanupSettings, TrashEmptyResult, TrashMoveResult, TrashStats } from './cleanup';
import type { PromptSlot } from './promptSlots';

export interface GenerationParams {
  prompt: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  /** Video families only: frame count (e.g. 81 frames @ 16fps ≈ 5s). */
  length?: number;
  /** Video families only: local path to the source image to animate, chosen via
   * window.kvgenius.chooseSourceImage() and uploaded to ComfyUI at generation time. */
  sourceImagePath?: string;
  /** Video upscale family only: local path to the video to enlarge. */
  sourceVideoPath?: string;
  /** Upscale family only: file name of the ComfyUI upscale model to use (see listUpscaleModels). */
  upscaleModel?: string;
}

export interface GenerationRecord {
  id: number;
  prompt: string;
  negativePrompt: string | null;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  length: number | null;
  modelFamily: string;
  imagePath: string;
  favorite: boolean;
  /** Kept out of the Library unless "Show hidden" is on - set by the hidden-words rule or by hand. */
  hidden: boolean;
  /** Marked as the representative example of its prompt - these make up Library > Prompts. */
  pinned: boolean;
  /** Video only: the app's kept copy of the image the video was made from, so Re-rack can re-run it
   * in place. Null for videos made before this was kept, and for everything that is not a video. */
  sourceImagePath: string | null;
  /** When it was moved to the Trash (ISO), or null for everything in the Library. */
  trashedAt: string | null;
  createdAt: string;
  /** How long this took vs what was predicted. Lives in its own table (see TimingStatRow) and is
   * null for generations made before timing was tracked. */
  timing: TimingInfo | null;
  /** Only in a Library listing grouped by prompt: this record is the cover of a stack, and this is how
   * many items share its exact prompt (itself included). */
  groupCount?: number;
  /** Only in a grouped listing: the newest item of the stack. Stacks are ordered, and paged, by this. */
  groupNewestId?: number;
  /** Only in a grouped listing, on a stack of more than one: the files of some of its items, the cover
   * first, for the stack's card to cycle through. */
  groupPreviewPaths?: string[];
}

/** Options for a Library listing beyond the usual filters. */
export interface LibraryListOptions {
  /** Collapse items whose prompt is exactly the same into one stack, shown by a cover item. */
  grouped?: boolean;
  /** Only the items whose prompt is exactly this (what opening a stack shows). */
  prompt?: string | null;
}

/** The estimate shown for a run and how long it actually took, in milliseconds. */
export interface TimingInfo {
  estimateMs: number | null;
  /** Estimate excluding model loading - comparable across cold and warm runs. */
  estimateGenerateMs: number | null;
  actualMs: number;
  generateMs: number | null;
  loadMs: number | null;
}

/** A prediction of how long a run will take, built from the timings of earlier runs. */
export interface TimeEstimate {
  /** 'exact': the same settings were run before; 'scaled': extrapolated from similar settings. */
  basis: 'exact' | 'scaled';
  samples: number;
  /** Expected model-load time; null when there is no history for this cold/warm situation. */
  loadMs: number | null;
  samplingMs: number;
  /** Decode + save after sampling. */
  finishMs: number;
  /** Typical time of one sampling step. */
  paceMs: number;
  /** sampling + finish. */
  generateMs: number;
  /** generate + load (when known). */
  totalMs: number;
}

export type GenerationPhase =
  | 'starting'
  | 'loading'
  | 'encoding'
  | 'preparing'
  | 'sampling'
  | 'decoding'
  | 'saving'
  | 'working';

/** Live progress of the running generation, pushed from the main process while ComfyUI works. */
export interface GenerationProgress {
  phase: GenerationPhase;
  /** Step within the current sampler (1-based), 0 before the first step reports. */
  step: number;
  stepMax: number;
  /** Which sampler pass is running (1-based) and how many the workflow has (video has two). */
  stage: number;
  stageCount: number;
  /** Sampling steps finished across all passes so far, and the total once every pass has reported. */
  stepsDone: number;
  stepsTotal: number | null;
  /** Milliseconds since the run started when the first sampling step reported (null before). */
  firstStepAtMs: number | null;
  /** Milliseconds since the run started. */
  elapsedMs: number;
}

/** One row of the timing-accuracy data. Deliberately holds nothing about the content - no prompt,
 * seed, image or file path - only timings and the settings that drive them. */
export interface TimingStatRow {
  id: number;
  createdAt: string;
  family: string;
  kind: GenerationKind;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  /** Video frames (null for images). */
  length: number | null;
  /** Whether the previous run used the same model family (models likely still loaded). */
  warm: boolean;
  estimateMs: number | null;
  estimateGenerateMs: number | null;
  actualMs: number;
  loadMs: number | null;
  generateMs: number | null;
  samplingMs: number | null;
  finishMs: number | null;
  samplerSteps: number | null;
  paceMs: number | null;
}

export type GenerationKind = 'image' | 'video';

/** Just enough of a generation to act on it (delete, export) without loading the whole record. */
export interface GenerationRef {
  id: number;
  imagePath: string;
  favorite: boolean;
  pinned: boolean;
}

export type ExportResult = { status: 'saved'; path: string; count: number } | { status: 'cancelled' };

/** A request to open the Generate tab in video mode with an existing image as the source. */
export interface VideoSourceRequest {
  imagePath: string;
  /** The image's own dimensions, used to pick a video size that keeps its aspect ratio. */
  width: number;
  height: number;
}

/** Which model families produce a video vs a still image - drives whether the
 * renderer shows an <img> or a <video> for a given record's output/result. */
export const FAMILY_KIND: Record<string, 'image' | 'video'> = {
  'z-image-turbo': 'image',
  'wan22-i2v': 'video',
  'upscale-video': 'video',
};

export interface GenerateResult {
  record: GenerationRecord;
  imageUrl: string;
}

export interface ComfyUIHostInfo {
  host: string;
  defaultHost: string;
}

export interface DbInfo {
  path: string;
  isDefault: boolean;
  defaultPath: string;
  /** Current database file's size in bytes, or null if the file doesn't exist yet. */
  sizeBytes: number | null;
}

export interface UpdateCheckResult {
  status: 'available' | 'not-available' | 'error' | 'unsupported';
  version?: string;
  message?: string;
}

/** What the "launch ComfyUI" shortcut would run: the program chosen in Settings, else one found in
 * ComfyUI Desktop's default install location. */
export interface ComfyUILauncherInfo {
  configured: string | null;
  detected: string | null;
}

export type ComfyUILaunchResult =
  | { status: 'launched' | 'already-running' | 'cancelled' }
  | { status: 'error'; message: string };

/** State of the local control API that MCP clients use, and of ffmpeg (needed for stitching videos). */
export interface McpInfo {
  enabled: boolean;
  running: boolean;
  port: number | null;
  /** Ready-to-paste `mcpServers` entry for an MCP client's config (e.g. Claude Desktop). */
  configSnippet: string;
  ffmpeg: { available: boolean; path: string | null; override: string | null };
}

/** Contract exposed on window.kvgenius by the preload script. */
export interface KVGeniusAPI {
  /** `estimate` is what was shown for this run; it is stored with the actual timing so the
   * accuracy of the estimates can be checked later. */
  generate: (
    family: string,
    params: GenerationParams,
    estimate?: { totalMs: number | null; generateMs: number | null } | null
  ) => Promise<GenerateResult>;
  /** Predicts how long a run will take from earlier runs' timings; null until there is history.
   * `previousFamily` is the family that will run just before it (null = none), which decides
   * whether the models are expected to be loaded already; omit it to use the last run. */
  estimateGeneration: (
    family: string,
    params: GenerationParams,
    previousFamily?: string | null
  ) => Promise<TimeEstimate | null>;
  /** Live stage/step updates for the generation in progress. Returns an unsubscribe function. */
  onGenerationProgress: (callback: (progress: GenerationProgress) => void) => () => void;
  /** Every recorded timing (estimate vs actual), newest first. Holds no prompt or image data. */
  getTimingStats: () => Promise<TimingStatRow[]>;
  clearTimingStats: () => Promise<void>;
  /** Interrupts the in-flight generation on ComfyUI's side (not just gives up waiting for it
   * client-side) and unblocks the pending generate() call. Safe to call with nothing in flight. */
  cancelGeneration: () => Promise<void>;
  /** One page of generations of the given kind, newest first: up to `limit` records with an id
   * below `beforeId` (null = start from the newest). Cursor-based so deletes between pages
   * can't skip or repeat rows. */
  listGenerations: (
    kind: GenerationKind,
    limit: number,
    beforeId: number | null,
    favoritesOnly: boolean,
    showHidden: boolean,
    /** Images only: show just this file type (e.g. 'gif'). */
    extension?: string | null,
    /** Grouping by prompt, or one prompt's items. A grouped page is cursored by the last record's `groupNewestId`. */
    options?: LibraryListOptions
  ) => Promise<GenerationRecord[]>;
  /** Totals per kind, restricted to favorites when `favoritesOnly`; hidden ones only count when `showHidden`.
   * `imageExtension` (e.g. 'gif') narrows the image count only. */
  countGenerations: (
    favoritesOnly: boolean,
    showHidden: boolean,
    imageExtension?: string | null,
    /** With `grouped`, counts stacks (distinct prompts) rather than items. */
    options?: LibraryListOptions
  ) => Promise<Record<GenerationKind, number>>;
  /** File extensions present among the Library's images (lowercase, no dot), most common first. */
  listImageExtensions: () => Promise<string[]>;
  /** Every generation of the kind (newest first), for Select All across pages that aren't loaded. */
  listGenerationRefs: (
    kind: GenerationKind,
    favoritesOnly: boolean,
    showHidden: boolean,
    extension?: string | null,
    /** Only `prompt` applies here: refs are always individual items, never stacks. */
    options?: LibraryListOptions
  ) => Promise<GenerationRef[]>;
  setGenerationHidden: (id: number, hidden: boolean) => Promise<void>;
  /** Words that, found in a prompt, mark the generation as hidden (see shared/hiddenWords.ts). */
  getHiddenWords: () => Promise<string[]>;
  /** Saves the list (cleaned up) and returns what was stored. Affects new generations only. */
  setHiddenWords: (words: string[]) => Promise<string[]>;
  /** Runs the saved word list over every existing generation, hiding the ones that match. It never
   * un-hides anything, so hand-hidden items stay hidden. */
  applyHiddenWords: () => Promise<{ checked: number; newlyHidden: number }>;
  /** Size of an output file in bytes, or null if it is missing. */
  getFileSize: (imagePath: string) => Promise<number | null>;
  /** Asks where to save, then writes the given output files into a zip archive there. */
  exportGenerations: (imagePaths: string[]) => Promise<ExportResult>;
  /** Favoriting moves the output file into a `favorites` subfolder (and unfavoriting moves it back),
   * so this returns the file's - possibly new - path. */
  setGenerationFavorite: (id: number, favorite: boolean) => Promise<{ imagePath: string }>;
  imageUrlFor: (imagePath: string) => string;
  /** Opens a native file dialog for picking a video mode's source image.
   * Resolves the chosen local path, or null if cancelled. */
  chooseSourceImage: () => Promise<string | null>;
  /** Like chooseSourceImage, but for any number of images (Tools > Upscale). Resolves the chosen
   * local paths, empty if cancelled. */
  chooseSourceImages: () => Promise<string[]>;
  /** The local paths of picture files dropped onto the window (pass the dropped `File` objects).
   * Resolves only the existing PNG / JPG / WebP ones, each now usable as a source image. */
  droppedImagePaths: (files: unknown[]) => Promise<string[]>;
  /** Puts an image on the clipboard as a picture (a GIF or other animation copies as one still frame).
   * Rejects for a file the app does not serve or cannot read as an image. */
  copyImageToClipboard: (imagePath: string) => Promise<void>;
  /** Reveals the generation's output file in the system file manager. */
  revealGenerationInFileManager: (imagePath: string) => Promise<void>;
  /** Explains why a video will not play (file layout, codec) and tries to repair it. */
  diagnoseVideo: (imagePath: string) => Promise<{ lines: string[]; repaired: boolean }>;
  /** Opens the output file in the system's default app (e.g. a video player). */
  openGenerationExternally: (imagePath: string) => Promise<void>;
  /** Opens a native save dialog and copies the output file to the chosen location.
   * Resolves true if saved, false if the dialog was cancelled. */
  saveGenerationAs: (imagePath: string) => Promise<boolean>;

  /** Pins a generation as a representative example of its prompt (Library > Prompts), or unpins it. */
  /** Pins or unpins. `groupSize` is how many pinned items now have this exact prompt (they share one
   * tile under Library > Prompts); 1 means this is the only one, 0 means it was unpinned. */
  setGenerationPinned: (id: number, pinned: boolean) => Promise<{ groupSize: number }>;
  /** An existing Library item that generating these settings again would only repeat, or null. */
  findDuplicateGeneration: (family: string, params: GenerationParams) => Promise<GenerationRecord | null>;
  /** Every pinned generation, most recently pinned first; hidden ones only with `showHidden`. */
  listPinnedGenerations: (showHidden: boolean) => Promise<GenerationRecord[]>;

  /** Deleting is two steps: items go to the Trash (no confirmation - they can be restored), and emptying
   * the Trash sends the files to the operating system's Recycle Bin. Both can also run on a schedule,
   * each its own option and both off until turned on. */
  getCleanupSettings: () => Promise<CleanupSettings>;
  setCleanupSettings: (
    patch: Partial<Pick<CleanupSettings, 'autoTrashEnabled' | 'olderThanDays' | 'autoEmptyEnabled' | 'trashRetentionDays'>>
  ) => Promise<CleanupSettings>;
  /** How many items (and bytes) a cleanup of items older than `days` would move: not favorited, not pinned. */
  previewCleanup: (days: number) => Promise<TrashStats>;
  /** Moves those items to the Trash. Never a favorite or pinned item. */
  runCleanup: (days: number) => Promise<TrashMoveResult>;
  /** Moves the given generations to the Trash. Favorites and pinned items are skipped unless
   * `includeKept` - set it only for something the user deleted item by item. */
  trashGenerations: (ids: number[], options?: { includeKept?: boolean }) => Promise<TrashMoveResult>;
  getTrashStats: () => Promise<TrashStats>;
  /** What is in the Trash, newest generation first: up to `limit` with an id below `beforeId`. */
  listTrashed: (limit: number, beforeId: number | null) => Promise<GenerationRecord[]>;
  restoreGenerations: (ids: number[]) => Promise<{ restored: number; failed: number }>;
  /** Removes items from the Trash: their files go to the Recycle Bin and they can no longer be restored
   * into the app. An item the Recycle Bin will not take stays in the Trash and counts as failed. */
  deleteTrashed: (ids: number[]) => Promise<TrashEmptyResult>;
  emptyTrash: () => Promise<TrashEmptyResult>;

  /** The Generate page's prompt "tabs" (whole form per tab), persisted across restarts. */
  getPromptSlots: () => Promise<{ slots: PromptSlot[]; activeId: string | null }>;
  savePromptSlots: (slots: PromptSlot[], activeId: string) => Promise<void>;

  getComfyUIHost: () => Promise<ComfyUIHostInfo>;
  setComfyUIHost: (host: string) => Promise<void>;
  resetComfyUIHost: () => Promise<void>;
  checkComfyUIConnection: () => Promise<boolean>;
  /** Starts ComfyUI if it is not already up. Asks which program to run the first time if none is
   * set or found ('cancelled' if that dialog is dismissed). Resolves once the program has started,
   * not once ComfyUI is ready - keep checking checkComfyUIConnection() for that. */
  launchComfyUI: () => Promise<ComfyUILaunchResult>;
  getComfyUILauncher: () => Promise<ComfyUILauncherInfo>;
  /** Opens a file dialog to choose the program to launch; resolves the new state, or null if cancelled. */
  chooseComfyUILauncher: () => Promise<ComfyUILauncherInfo | null>;
  clearComfyUILauncher: () => Promise<ComfyUILauncherInfo>;

  getTheme: () => Promise<string>;
  setTheme: (themeId: string) => Promise<void>;

  getDbInfo: () => Promise<DbInfo>;
  revealDbInFileManager: () => Promise<void>;
  /** Each resolves the chosen path, or null if the dialog was cancelled. Choosing a path
   * relaunches the app (a live database connection can't be repointed at a new file). */
  chooseExistingDb: () => Promise<string | null>;
  chooseNewDbLocation: () => Promise<string | null>;
  resetDbToDefault: () => Promise<void>;

  getMcpInfo: () => Promise<McpInfo>;
  /** Turns the local control API on or off (off by default). */
  setMcpEnabled: (enabled: boolean) => Promise<McpInfo>;
  /** Asks for an ffmpeg binary; resolves the new state, or null if the dialog was cancelled. */
  chooseFfmpegPath: () => Promise<McpInfo | null>;
  resetFfmpegPath: () => Promise<McpInfo>;

  getAppVersion: () => Promise<string>;
  checkForUpdates: () => Promise<UpdateCheckResult>;

  /** Launch or focus the Hardpoint AI services dashboard. */
  openHardpoint: () => Promise<{ status: 'ok' } | { status: 'error'; message: string }>;
  /** Main-process probe of Hardpoint's loopback API (renderer fetch is blocked). */
  hardpointIsReachable: () => Promise<boolean>;

  /** Upscale models installed in ComfyUI (its models/upscale_models folder). Rejects if ComfyUI is unreachable. */
  listUpscaleModels: () => Promise<string[]>;

  /** Converts a Library video to a GIF (ffmpeg, no audio) and saves it as a new Library image. */
  convertToGif: (id: number, options: { fps: number; width: number }) => Promise<GenerateResult>;
}
