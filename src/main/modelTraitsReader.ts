import * as fs from 'fs';
import * as path from 'path';
import { MODEL_FOLDERS } from '../shared/modelManifest';
import { ModelTraits, readTraits } from '../shared/modelTraits';
import { readSafetensorsHeader } from './safetensors';

/** Traits of files already read, by path and the size / modified time they were read at, so an unchanged file is read once. */
const cache = new Map<string, ModelTraits | null>();
const CACHE_LIMIT = 2000;

/** The traits of one file (see shared/modelTraits.ts), or null when it is not a readable .safetensors file. Reads only its header. */
export async function readFileTraits(filePath: string): Promise<ModelTraits | null> {
  if (path.extname(filePath).toLowerCase() !== '.safetensors') return null;
  let stat: fs.Stats;
  try {
    stat = await fs.promises.stat(filePath);
  } catch {
    return null;
  }
  const key = `${filePath}|${stat.size}|${stat.mtimeMs}`;
  if (cache.has(key)) return cache.get(key) ?? null;
  const read = await readSafetensorsHeader(filePath);
  const traits = read.ok ? readTraits(Object.fromEntries(Object.entries(read.header.tensors).map(([name, info]) => [name, info.shape]))) : null;
  if (cache.size >= CACHE_LIMIT) cache.clear();
  cache.set(key, traits);
  return traits;
}

/** How many headers are read at once. */
const PARALLEL = 8;

/**
 * Older folder names ComfyUI still searches for a kind of model (its `unet` folder is where diffusion models used to go, `clip` where
 * text encoders did). A file it lists from there is in a different directory from the folder the app is named after.
 */
const LEGACY_FOLDER_NAMES: Partial<Record<(typeof MODEL_FOLDERS)[number], string[]>> = {
  diffusion_models: ['unet'],
  text_encoders: ['clip'],
};

/** Where the files of a models folder can be: the folder itself and its legacy names under the models folder. */
export function folderRoots(modelsDir: string, folder: string): string[] {
  if (!(MODEL_FOLDERS as readonly string[]).includes(folder)) return [];
  const names = [folder, ...(LEGACY_FOLDER_NAMES[folder as (typeof MODEL_FOLDERS)[number]] ?? [])];
  return names.map((name) => path.resolve(modelsDir, name));
}

/**
 * The traits of the named files, each looked for in `roots` in order (the first directory that has it wins, as in ComfyUI, which lists
 * a file once however many of its folders hold it). Names are as ComfyUI lists them, so possibly "sub/file.safetensors". A name that
 * would leave its directory, or is in none of them, is answered with null rather than read - the names come from the renderer.
 */
export async function readTraitsInRoots(roots: string[], files: string[]): Promise<Record<string, ModelTraits | null>> {
  const result: Record<string, ModelTraits | null> = {};
  const resolvedRoots = roots.filter((r) => typeof r === 'string' && path.isAbsolute(r)).map((r) => path.resolve(r));
  const queue = files.filter((f): f is string => typeof f === 'string').slice(0, 5000);
  let next = 0;
  async function worker() {
    while (next < queue.length) {
      const name = queue[next++];
      result[name] = null;
      for (const root of resolvedRoots) {
        const full = path.resolve(root, name);
        if (!full.startsWith(root + path.sep)) continue;
        try {
          if (!(await fs.promises.stat(full)).isFile()) continue;
        } catch {
          continue;
        }
        result[name] = await readFileTraits(full);
        break;
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(PARALLEL, queue.length) }, worker));
  return result;
}

/** The traits of the named files of one models sub-folder, looked for under `modelsDir` (including the folder's legacy names). */
export async function readFolderTraits(modelsDir: string, folder: string, files: string[]): Promise<Record<string, ModelTraits | null>> {
  const roots = folderRoots(modelsDir, folder);
  return roots.length === 0 ? {} : readTraitsInRoots(roots, files);
}
