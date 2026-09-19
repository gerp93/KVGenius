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

/** Contract exposed on window.kvgenius by the preload script. */
export interface KVGeniusAPI {
  generate: (params: GenerationParams) => Promise<GenerateResult>;
  listGenerations: () => Promise<GenerationRecord[]>;
  imageUrlFor: (imagePath: string) => string;

  listSavedPrompts: () => Promise<SavedPrompt[]>;
  savePrompt: (name: string | null, prompt: string) => Promise<SavedPrompt>;
  deleteSavedPrompt: (id: number) => Promise<void>;
}
