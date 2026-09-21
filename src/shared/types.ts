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
  createdAt: string;
}

export type GenerationKind = 'image' | 'video';

/** Just enough of a generation to act on it (delete, export) without loading the whole record. */
export interface GenerationRef {
  id: number;
  imagePath: string;
  favorite: boolean;
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
};

export interface SavedPrompt {
  id: number;
  name: string | null;
  prompt: string;
  negativePrompt: string | null;
  createdAt: string;
}

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
}

export interface UpdateCheckResult {
  status: 'available' | 'not-available' | 'error' | 'unsupported';
  version?: string;
  message?: string;
}

/** Contract exposed on window.kvgenius by the preload script. */
export interface KVGeniusAPI {
  generate: (family: string, params: GenerationParams) => Promise<GenerateResult>;
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
    favoritesOnly: boolean
  ) => Promise<GenerationRecord[]>;
  /** Totals per kind, restricted to favorites when `favoritesOnly`. */
  countGenerations: (favoritesOnly: boolean) => Promise<Record<GenerationKind, number>>;
  /** Every generation of the kind (newest first), for Select All across pages that aren't loaded. */
  listGenerationRefs: (kind: GenerationKind, favoritesOnly: boolean) => Promise<GenerationRef[]>;
  /** Size of an output file in bytes, or null if it is missing. */
  getFileSize: (imagePath: string) => Promise<number | null>;
  /** Asks where to save, then writes the given output files into a zip archive there. */
  exportGenerations: (imagePaths: string[]) => Promise<ExportResult>;
  setGenerationFavorite: (id: number, favorite: boolean) => Promise<void>;
  imageUrlFor: (imagePath: string) => string;
  /** Opens a native file dialog for picking a video mode's source image.
   * Resolves the chosen local path, or null if cancelled. */
  chooseSourceImage: () => Promise<string | null>;
  /** Deletes the generation's DB row and its output file on disk. */
  deleteGeneration: (id: number, imagePath: string) => Promise<void>;
  /** Reveals the generation's output file in the system file manager. */
  revealGenerationInFileManager: (imagePath: string) => Promise<void>;
  /** Explains why a video will not play (file layout, codec) and tries to repair it. */
  diagnoseVideo: (imagePath: string) => Promise<{ lines: string[]; repaired: boolean }>;
  /** Opens the output file in the system's default app (e.g. a video player). */
  openGenerationExternally: (imagePath: string) => Promise<void>;
  /** Opens a native save dialog and copies the output file to the chosen location.
   * Resolves true if saved, false if the dialog was cancelled. */
  saveGenerationAs: (imagePath: string) => Promise<boolean>;

  listSavedPrompts: () => Promise<SavedPrompt[]>;
  savePrompt: (name: string | null, prompt: string) => Promise<SavedPrompt>;
  deleteSavedPrompt: (id: number) => Promise<void>;

  getComfyUIHost: () => Promise<ComfyUIHostInfo>;
  setComfyUIHost: (host: string) => Promise<void>;
  resetComfyUIHost: () => Promise<void>;
  checkComfyUIConnection: () => Promise<boolean>;

  getTheme: () => Promise<string>;
  setTheme: (themeId: string) => Promise<void>;

  getDbInfo: () => Promise<DbInfo>;
  revealDbInFileManager: () => Promise<void>;
  /** Each resolves the chosen path, or null if the dialog was cancelled. Choosing a path
   * relaunches the app (a live database connection can't be repointed at a new file). */
  chooseExistingDb: () => Promise<string | null>;
  chooseNewDbLocation: () => Promise<string | null>;
  resetDbToDefault: () => Promise<void>;

  getAppVersion: () => Promise<string>;
  checkForUpdates: () => Promise<UpdateCheckResult>;

  /** Launch or focus the Hardpoint AI services dashboard. */
  openHardpoint: () => Promise<{ status: 'ok' } | { status: 'error'; message: string }>;
  /** Main-process probe of Hardpoint's loopback API (renderer fetch is blocked). */
  hardpointIsReachable: () => Promise<boolean>;
}
