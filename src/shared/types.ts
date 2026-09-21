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
  createdAt: string;
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
  listGenerations: () => Promise<GenerationRecord[]>;
  imageUrlFor: (imagePath: string) => string;
  /** Opens a native file dialog for picking a video mode's source image.
   * Resolves the chosen local path, or null if cancelled. */
  chooseSourceImage: () => Promise<string | null>;
  /** Deletes the generation's DB row and its output file on disk. */
  deleteGeneration: (id: number, imagePath: string) => Promise<void>;
  /** Reveals the generation's output file in the system file manager. */
  revealGenerationInFileManager: (imagePath: string) => Promise<void>;
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
