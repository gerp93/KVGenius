import * as fs from 'fs';
import * as path from 'path';
import { Transform } from 'stream';
import { pipeline } from 'stream/promises';
import { MODEL_FOLDERS, ModelFolder } from '../shared/modelManifest';
import { isModelFileName, modelFileExtension, ModelFileCheck } from '../shared/modelCheck';
import { compareHeaders, readSafetensorsHeader } from './safetensors';

type Finding = { level: 'block' | 'warn' | 'info'; text: string };

export interface CheckOptions {
  /** The folder the file would go in - decides how strict to be about the older, code-carrying formats. */
  folder: ModelFolder;
  /** ComfyUI's models folder, if known. */
  modelsDir: string | null;
  /** The slot's known-good file (the one the shipped template loads), to compare layouts with. */
  referenceFile: string | null;
}

const MB = 1024 * 1024;

function format(bytes: number): string {
  return bytes >= 1024 * MB ? `${(bytes / (1024 * MB)).toFixed(1)} GB` : `${(bytes / MB).toFixed(0)} MB`;
}

/**
 * Looks a model file over before anything is copied: is it a model file at all, is it all there (a safetensors
 * header says exactly how big the file must be, so a cut-off download is caught), and does its tensor layout
 * match the known-good file for the slot. Reads only headers, so it is quick however big the file is.
 */
export async function checkModelFile(srcPath: string, options: CheckOptions): Promise<ModelFileCheck> {
  const fileName = path.basename(srcPath);
  const ext = modelFileExtension(fileName);
  const findings: Finding[] = [];
  let sizeBytes = 0;
  let comparison: ModelFileCheck['comparison'] = null;

  const finish = (): ModelFileCheck => {
    const severity = findings.some((f) => f.level === 'block') ? 'block' : findings.some((f) => f.level === 'warn') ? 'warn' : 'ok';
    const order = { block: 0, warn: 1, info: 2 } as const;
    return { fileName, sizeBytes, severity, messages: [...findings].sort((a, b) => order[a.level] - order[b.level]).map((f) => f.text), comparison };
  };

  try {
    const stat = fs.statSync(srcPath);
    if (!stat.isFile()) throw new Error('not a file');
    sizeBytes = stat.size;
  } catch {
    findings.push({ level: 'block', text: 'The file cannot be read.' });
    return finish();
  }

  if (!isModelFileName(fileName)) {
    findings.push({ level: 'block', text: `"${ext || fileName}" is not a model file - expected .safetensors.` });
    return finish();
  }
  if (ext === '.gguf') {
    findings.push({ level: 'block', text: 'GGUF files need extra nodes installed in ComfyUI and are not supported here.' });
    return finish();
  }
  if (ext !== '.safetensors') {
    if (options.folder !== 'upscale_models') {
      findings.push({ level: 'warn', text: `${ext} files can run code when loaded - only use one from a source you trust. A .safetensors copy is safer if one exists.` });
    }
    findings.push({ level: 'info', text: 'This format cannot be looked inside, so only a test render can tell whether it fits.' });
    return finish();
  }

  const read = await readSafetensorsHeader(srcPath);
  if (!read.ok) {
    findings.push({ level: 'block', text: read.reason });
    return finish();
  }
  if (read.fileSize < read.header.expectedSize) {
    findings.push({
      level: 'block',
      text: `The file is incomplete: it is ${format(read.fileSize)} but should be ${format(read.header.expectedSize)}. The download was probably cut off - download it again.`,
    });
    return finish();
  }

  const reference = options.modelsDir && options.referenceFile ? path.join(options.modelsDir, options.folder, options.referenceFile) : null;
  if (!reference || !fs.existsSync(reference)) {
    findings.push({ level: 'info', text: 'Not compared with a known-good file (that needs the models folder set and the original file in it), so only a test render can tell whether it fits.' });
    return finish();
  }
  const ref = await readSafetensorsHeader(reference);
  if (!ref.ok) {
    findings.push({ level: 'info', text: `Could not read ${options.referenceFile} to compare with.` });
    return finish();
  }
  comparison = compareHeaders(ref.header, read.header);
  const name = options.referenceFile as string;
  if (comparison.verdict === 'match') {
    findings.push({ level: 'info', text: `Same layout as ${name} (${comparison.shared} tensors) - a good sign it fits.` });
  } else if (comparison.verdict === 'related') {
    findings.push({
      level: 'warn',
      text: `Similar to ${name} but not the same: ${comparison.shared} of ${comparison.referenceTensors} tensors match${comparison.shapeMismatches ? `, ${comparison.shapeMismatches} with a different size` : ''}. It may be a different version - try a test render.`,
    });
  } else {
    findings.push({
      level: 'warn',
      text: `Does not look like ${name}: almost none of its ${comparison.referenceTensors} tensors are in this file. It is probably a different kind of model, or the same one packaged another way (an all-in-one checkpoint, say).`,
    });
  }
  return finish();
}

export class ModelImportError extends Error {
  constructor(
    readonly code: 'exists' | 'no-space' | 'bad-target' | 'cancelled' | 'incomplete',
    message: string
  ) {
    super(message);
  }
}

export interface ImportOptions {
  modelsDir: string;
  folder: ModelFolder;
  /** Remove the original once the copy is verified. */
  move?: boolean;
  /** Replace a file of the same name already there. */
  overwrite?: boolean;
  onProgress?: (copied: number, total: number) => void;
  signal?: AbortSignal;
}

export interface ImportResult {
  destPath: string;
  bytes: number;
  /** Move asked for, but the original could not be removed (the copy is still good). */
  originalKept: boolean;
}

async function freeBytes(dir: string): Promise<number | null> {
  try {
    const s = await fs.promises.statfs(dir);
    return Number(s.bavail) * Number(s.bsize);
  } catch {
    return null;
  }
}

/**
 * Copies (or moves) a model file into `<modelsDir>/<folder>/`, keeping its name - the name is how the
 * workflow finds it. Written as `<name>.part` and renamed only when whole, so ComfyUI never sees half a file and
 * a cancel leaves nothing behind. Refuses to overwrite unless told to, and checks free space first.
 */
export async function importModelFile(srcPath: string, options: ImportOptions): Promise<ImportResult> {
  const { modelsDir, folder, signal } = options;
  if (!(MODEL_FOLDERS as readonly string[]).includes(folder)) throw new ModelImportError('bad-target', `"${folder}" is not a model folder.`);
  if (!fs.existsSync(modelsDir) || !fs.statSync(modelsDir).isDirectory()) throw new ModelImportError('bad-target', "ComfyUI's models folder is not set or does not exist.");
  const fileName = path.basename(srcPath);
  const destDir = path.join(modelsDir, folder);
  const destPath = path.join(destDir, fileName);
  const inside = path.relative(destDir, destPath);
  if (!isModelFileName(fileName) || inside !== fileName) throw new ModelImportError('bad-target', `"${fileName}" cannot be imported.`);

  const total = fs.statSync(srcPath).size;
  if (path.resolve(srcPath) === path.resolve(destPath)) return { destPath, bytes: total, originalKept: false };
  if (fs.existsSync(destPath) && !options.overwrite) throw new ModelImportError('exists', `${fileName} is already in ${folder}.`);

  fs.mkdirSync(destDir, { recursive: true });
  const free = await freeBytes(destDir);
  if (free !== null && free < total) {
    throw new ModelImportError('no-space', `Not enough free space: ${format(total)} needed, ${format(free)} free.`);
  }

  const part = `${destPath}.part`;
  let copied = 0;
  let lastReport = 0;
  const counter = new Transform({
    transform(chunk: Buffer, _enc, callback) {
      copied += chunk.length;
      const now = Date.now();
      if (options.onProgress && now - lastReport >= 200) {
        lastReport = now;
        options.onProgress(copied, total);
      }
      callback(null, chunk);
    },
  });
  try {
    const streams = [fs.createReadStream(srcPath), counter, fs.createWriteStream(part)] as const;
    if (signal) await pipeline(...streams, { signal });
    else await pipeline(...streams);
    if (fs.statSync(part).size !== total) throw new ModelImportError('incomplete', 'The copy is not the same size as the original.');
    if (options.overwrite) fs.rmSync(destPath, { force: true });
    fs.renameSync(part, destPath);
  } catch (err) {
    fs.rmSync(part, { force: true });
    if (signal?.aborted) throw new ModelImportError('cancelled', 'Import cancelled.');
    throw err;
  }
  options.onProgress?.(total, total);

  let originalKept = false;
  if (options.move) {
    try {
      fs.rmSync(srcPath);
    } catch {
      originalKept = true;
    }
  }
  return { destPath, bytes: total, originalKept };
}
