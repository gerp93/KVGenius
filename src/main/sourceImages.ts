import { DatabaseSync } from 'node:sqlite';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

/**
 * A video is made from a source image that is only ever uploaded to ComfyUI, so without a copy a
 * recalled video could not be re-run in place: the original may be anywhere on disk (outside what the
 * app may show), or inside the Library where favoriting moves it and deleting removes it. So each
 * video keeps its own copy in the app's sources folder, named by a hash of its contents - the same
 * picture used for several videos is stored once.
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

/** Deletes a kept source image once no generation refers to it any more. Only ever touches files
 * inside `sourcesDir`, so a path that points anywhere else (an old record, a hand-edited row) is left
 * alone. Returns whether a file was deleted. */
export function releaseSourceImage(db: DatabaseSync, sourcePath: string | null, sourcesDir: string): boolean {
  if (!sourcePath || !isInside(sourcePath, sourcesDir)) return false;
  const stillUsed = db.prepare('SELECT 1 FROM generations WHERE source_image_path = ? LIMIT 1').get(sourcePath);
  if (stillUsed) return false;
  try {
    fs.unlinkSync(sourcePath);
    return true;
  } catch {
    return false;
  }
}
