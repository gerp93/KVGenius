/** How a candidate model file's tensors compare with the known-good file for the same slot. */
export interface TensorComparison {
  verdict: 'match' | 'related' | 'different';
  /** Tensors the known-good file has (ignoring quantisation bookkeeping). */
  referenceTensors: number;
  candidateTensors: number;
  /** Tensors present in both. */
  shared: number;
  /** Shared tensors whose shape differs. */
  shapeMismatches: number;
}

/** Result of looking a model file over before it is used or copied anywhere. */
export interface ModelFileCheck {
  fileName: string;
  sizeBytes: number;
  /** 'block': cannot be used. 'warn': may be wrong - the user decides. 'ok': nothing found wrong. */
  severity: 'ok' | 'warn' | 'block';
  /** Said in plain words, most important first. */
  messages: string[];
  comparison: TensorComparison | null;
}

/** Extensions a model slot accepts, and which of them can be looked inside. */
export const MODEL_FILE_EXTENSIONS = ['.safetensors', '.pth', '.pt', '.ckpt', '.gguf'] as const;

export function modelFileExtension(fileName: string): string {
  const i = fileName.lastIndexOf('.');
  return i < 0 ? '' : fileName.slice(i).toLowerCase();
}

export function isModelFileName(fileName: string): boolean {
  return (MODEL_FILE_EXTENSIONS as readonly string[]).includes(modelFileExtension(fileName));
}

/** A model file the user chose (by dialog or drop), allowed to be checked and imported. */
export interface PickedModelFile {
  path: string;
  fileName: string;
}

/** What importing a model file came to. */
export type ModelImportOutcome =
  | { ok: true; fileName: string; bytes: number; originalKept: boolean }
  | { ok: false; code: 'exists' | 'no-space' | 'bad-target' | 'cancelled' | 'incomplete' | 'failed'; message: string };

export interface ModelImportProgress {
  copied: number;
  total: number;
}

/** What could be read about how a picture was made. Only what the picture itself says - nothing is guessed. */
export interface ImageSettings {
  steps?: number;
  cfg?: number;
  sampler?: string;
  scheduler?: string;
  /** ModelSamplingAuraFlow / SD3 shift. */
  shift?: number;
  /** The loader files the workflow named, so a matching installed file can be suggested. */
  fileHints?: { diffusionModel?: string; textEncoder?: string; vae?: string };
  source: 'comfyui' | 'a1111';
}

/** The picture the user chose to read settings from, and what it said (null: nothing usable in it). */
export interface ImageSettingsResult {
  fileName: string;
  settings: ImageSettings | null;
}

export interface ModelTestResult {
  ok: boolean;
  /** Plain words for the user: what happened, or why it did not work. */
  message: string;
  /** The test picture, when there is one. */
  imageBase64?: string;
  mime?: string;
}
