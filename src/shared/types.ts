export interface GenerationParams {
  prompt: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
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
  modelFamily: string;
  imagePath: string;
  createdAt: string;
}

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
  generate: (params: GenerationParams) => Promise<GenerateResult>;
  listGenerations: () => Promise<GenerationRecord[]>;
  imageUrlFor: (imagePath: string) => string;

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
}
