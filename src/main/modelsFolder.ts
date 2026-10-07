import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { MODEL_FOLDERS } from '../shared/modelManifest';
import { emptyInstalled, InstalledModels } from '../shared/modelStatus';

/** Subfolders a real ComfyUI `models` folder has (newer and older names both count). */
const KNOWN_SUBFOLDERS = [...MODEL_FOLDERS, 'checkpoints', 'clip', 'unet', 'controlnet', 'embeddings'];

interface FsOptions {
  isDir?: (p: string) => boolean;
}

function isDirectory(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
}

/** True when the folder exists and looks like ComfyUI's `models` folder (has some of its usual subfolders). */
export function looksLikeModelsDir(dir: string, options: FsOptions = {}): boolean {
  const isDir = options.isDir ?? isDirectory;
  if (!isDir(dir)) return false;
  return KNOWN_SUBFOLDERS.some((sub) => isDir(path.join(dir, sub)));
}

interface GuessOptions extends FsOptions {
  home?: string;
}

/**
 * Where the models folder probably is, from the program that launches ComfyUI. A portable install's
 * run script sits beside `ComfyUI/models`; a script inside the ComfyUI folder sits beside `models`.
 * ComfyUI Desktop's launcher is the app itself and says nothing about it - the base folder is chosen at
 * install (believed to default to Documents/ComfyUI, which is only tried as a last guess). A guess is
 * only returned if it passes looksLikeModelsDir, so a wrong one is simply not offered.
 */
export function guessModelsDir(launchPath: string | null, options: GuessOptions = {}): string | null {
  const home = options.home ?? os.homedir();
  const candidates: string[] = [];
  if (launchPath) {
    const dir = path.dirname(launchPath);
    candidates.push(path.join(dir, 'ComfyUI', 'models'), path.join(dir, 'models'), path.join(dir, '..', 'models'));
  }
  candidates.push(path.join(home, 'Documents', 'ComfyUI', 'models'), path.join(home, 'ComfyUI', 'models'));
  return candidates.map((c) => path.normalize(c)).find((c) => looksLikeModelsDir(c, options)) ?? null;
}

/** Every model file under the folders we care about, as names relative to each (with "/" for subfolders). */
export function scanModelsDir(modelsDir: string): InstalledModels {
  const installed = emptyInstalled();
  for (const folder of MODEL_FOLDERS) {
    const root = path.join(modelsDir, folder);
    const walk = (dir: string, prefix: string) => {
      let entries: fs.Dirent[];
      try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const entry of entries) {
        if (entry.isDirectory()) walk(path.join(dir, entry.name), `${prefix}${entry.name}/`);
        else if (/\.(safetensors|pth|pt|ckpt|gguf|bin)$/i.test(entry.name)) installed[folder].push(`${prefix}${entry.name}`);
      }
    };
    walk(root, '');
    installed[folder].sort();
  }
  return installed;
}
