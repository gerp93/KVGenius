import { MODEL_FOLDERS, ModelFolder, ManifestFile } from './modelManifest';

/** The files ComfyUI (or a scan of the models folder) reports, per folder. Names use "/" for subfolders. */
export type InstalledModels = Record<ModelFolder, string[]>;

export function emptyInstalled(): InstalledModels {
  return Object.fromEntries(MODEL_FOLDERS.map((folder) => [folder, [] as string[]])) as unknown as InstalledModels;
}

/** Where the list came from: ComfyUI itself, a scan of the models folder (ComfyUI not running), or nowhere. */
export type ModelStatusSource = 'comfyui' | 'folder' | 'none';

export interface ModelStatusReport {
  source: ModelStatusSource;
  installed: InstalledModels;
}

/**
 * ComfyUI describes a loader's choice list as `[[...names]]` (older versions) or
 * `["COMBO", { options: [...] }]` (newer). Anything else yields no names.
 */
export function parseChoiceList(spec: unknown): string[] {
  if (!Array.isArray(spec)) return [];
  const choices = Array.isArray(spec[0]) ? spec[0] : (spec[1] as { options?: unknown } | undefined)?.options;
  return Array.isArray(choices) ? choices.map(String) : [];
}

export type FileState =
  /** Listed under exactly the name the template asks for. */
  | { state: 'present' }
  /** Not under that name, but a file with the same name sits in a subfolder, where the template will not find it. */
  | { state: 'in-subfolder'; foundAs: string }
  | { state: 'missing' }
  /** Nothing is known (ComfyUI is not reachable and no models folder is set). */
  | { state: 'unknown' };

function baseName(name: string): string {
  return name.split(/[\\/]/).pop() ?? name;
}

export function fileState(file: ManifestFile, report: ModelStatusReport): FileState {
  if (report.source === 'none') return { state: 'unknown' };
  const names = report.installed[file.folder] ?? [];
  if (names.includes(file.file)) return { state: 'present' };
  const nested = names.find((name) => baseName(name) === file.file);
  return nested ? { state: 'in-subfolder', foundAs: nested } : { state: 'missing' };
}

export interface FeatureSummary {
  total: number;
  present: number;
  missing: number;
  inSubfolder: number;
}

export function summarize(files: ManifestFile[], report: ModelStatusReport): FeatureSummary {
  const summary: FeatureSummary = { total: files.length, present: 0, missing: 0, inSubfolder: 0 };
  for (const file of files) {
    const s = fileState(file, report).state;
    if (s === 'present') summary.present++;
    else if (s === 'in-subfolder') summary.inSubfolder++;
    else if (s === 'missing') summary.missing++;
  }
  return summary;
}
