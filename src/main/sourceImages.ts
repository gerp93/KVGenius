import { DatabaseSync } from 'node:sqlite';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { INPAINT_FAMILY } from '../shared/imageToImage';
import { needsSourceImage } from '../shared/sourceFamilies';
import { V2V_FAMILY } from '../shared/videoToVideo';

/**
 * A video is made from a source image that is only ever uploaded to ComfyUI, so without a copy a
 * recalled video could not be re-run in place: the original may be anywhere on disk (outside what the
 * app may show), or inside the Library where favoriting moves it and deleting removes it. So each
 * video keeps its own copy in the app's sources folder, named by a hash of its contents - the same
 * picture used for several videos is stored once.
 *
 * The copy is made when the job is queued (see `keepJobSources`), not when it finishes: a job waiting in the queue must not
 * depend on the original still being where it was.
 */

function isInside(file: string, dir: string): boolean {
  const resolvedDir = path.resolve(dir);
  const resolved = path.resolve(file);
  const [a, b] = process.platform === 'win32' ? [resolved.toLowerCase(), resolvedDir.toLowerCase()] : [resolved, resolvedDir];
  return a.startsWith(b + path.sep);
}

/** Copies `sourcePath` into `sourcesDir` (unless an identical file is already there) and returns the
 * copy's path, or null if it could not be kept (the original is gone or unreadable). A file that is
 * already in `sourcesDir` is its own copy. */
export function keepSourceImage(sourcePath: string, sourcesDir: string): string | null {
  try {
    if (isInside(sourcePath, sourcesDir)) return fs.existsSync(sourcePath) ? path.resolve(sourcePath) : null;
    const bytes = fs.readFileSync(sourcePath);
    const hash = crypto.createHash('sha1').update(bytes).digest('hex').slice(0, 20);
    const target = path.join(sourcesDir, `${hash}${path.extname(sourcePath).toLowerCase()}`);
    if (!fs.existsSync(target)) {
      fs.mkdirSync(sourcesDir, { recursive: true });
      fs.writeFileSync(target, bytes);
    }
    return target;
  } catch {
    return null;
  }
}

/** Whether a job still waiting or running will use this file (a job's params are JSON, so the path appears escaped in them). */
function usedByPendingJob(db: DatabaseSync, filePath: string): boolean {
  try {
    const escaped = JSON.stringify(filePath).slice(1, -1);
    return !!db.prepare("SELECT 1 FROM jobs WHERE status IN ('queued', 'running') AND instr(params, ?) > 0 LIMIT 1").get(escaped);
  } catch {
    // No jobs table (a bare test database): nothing is pending.
    return false;
  }
}

/** Deletes a kept source image once no generation refers to it any more. Only ever touches files
 * inside `sourcesDir`, so a path that points anywhere else (an old record, a hand-edited row) is left
 * alone. Returns whether a file was deleted. */
export function releaseSourceImage(db: DatabaseSync, sourcePath: string | null, sourcesDir: string): boolean {
  if (!sourcePath || !isInside(sourcePath, sourcesDir)) return false;
  // A kept mask lives in the same folder, so a file is still wanted while it is any result's source image or mask.
  const stillUsed = db.prepare('SELECT 1 FROM generations WHERE source_image_path = ? OR mask_image_path = ? LIMIT 1').get(sourcePath, sourcePath);
  if (stillUsed) return false;
  if (usedByPendingJob(db, sourcePath)) return false;
  try {
    fs.unlinkSync(sourcePath);
    return true;
  } catch {
    return false;
  }
}

/** The pictures a job is made from that the app keeps a copy of: a source image (for the families that work from one) and an inpainting mask. */
export function keptFilesOfParams(family: string, params: { sourceImagePath?: string; maskImagePath?: string }): string[] {
  return [needsSourceImage(family) ? params.sourceImagePath : undefined, family === INPAINT_FAMILY ? params.maskImagePath : undefined].filter(
    (p): p is string => !!p
  );
}

/**
 * Copies a job's source image (and mask) into `sourcesDir` now and points the job at the copies, so it runs from them - the original
 * may move, be favorited into another folder or be deleted while the job waits. Throws, naming the problem, when an original is
 * already gone: better said at once than as a failed job later.
 */
export function keepJobSources<P extends { sourceImagePath?: string; maskImagePath?: string; sourceVideoPath?: string }>(family: string, params: P, sourcesDir: string): P {
  const next = { ...params };
  // A video to video source is a Library video, read in place (videos are too big to copy), so only check it is still there.
  if (family === V2V_FAMILY && params.sourceVideoPath && !fs.existsSync(params.sourceVideoPath)) {
    throw new Error(`The source video could not be found any more (${params.sourceVideoPath}). Choose it again.`);
  }
  if (needsSourceImage(family) && params.sourceImagePath) {
    const kept = keepSourceImage(params.sourceImagePath, sourcesDir);
    if (!kept) throw new Error(`The source image could not be found any more (${params.sourceImagePath}). Choose it again.`);
    next.sourceImagePath = kept;
  }
  if (family === INPAINT_FAMILY && params.maskImagePath) {
    const kept = keepSourceImage(params.maskImagePath, sourcesDir);
    if (!kept) throw new Error('The painted mask could not be found any more. Paint it again.');
    next.maskImagePath = kept;
  }
  return next;
}
