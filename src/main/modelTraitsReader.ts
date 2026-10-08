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
 * The traits of the named files of one models sub-folder (names as ComfyUI lists them, so possibly "sub/file.safetensors").
 * Anything that would leave the folder is answered with null rather than read - the names come from the renderer.
 */
export async function readFolderTraits(modelsDir: string, folder: string, files: string[]): Promise<Record<string, ModelTraits | null>> {
  const result: Record<string, ModelTraits | null> = {};
  if (!(MODEL_FOLDERS as readonly string[]).includes(folder)) return result;
  const root = path.resolve(modelsDir, folder);
  const queue = files.filter((f): f is string => typeof f === 'string').slice(0, 5000);
  let next = 0;
  async function worker() {
    while (next < queue.length) {
      const name = queue[next++];
      const full = path.resolve(root, name);
      result[name] = full.startsWith(root + path.sep) ? await readFileTraits(full) : null;
    }
  }
  await Promise.all(Array.from({ length: Math.min(PARALLEL, queue.length) }, worker));
  return result;
}
