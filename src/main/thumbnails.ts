import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

/** Pictures a Library card can show as a small copy: the still formats (a GIF is animated, so it is served as it is). */
const THUMBNAIL_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);

export const MIN_THUMBNAIL_WIDTH = 64;
export const MAX_THUMBNAIL_WIDTH = 2048;

export interface Thumbnail {
  data: Buffer;
  mime: string;
}

/** Makes a JPEG of `file` no wider than `width`, or null when it is already that small or cannot be read (Electron's nativeImage in the app). */
export type Resizer = (file: string, width: number) => Promise<Buffer | null>;

/** The width asked for in a `?w=` query, if it is one that may be made. */
export function thumbnailWidth(query: string | null): number | null {
  if (query === null || !/^\d{1,5}$/.test(query)) return null;
  const width = Number(query);
  return width >= MIN_THUMBNAIL_WIDTH && width <= MAX_THUMBNAIL_WIDTH ? width : null;
}

export function canThumbnail(file: string): boolean {
  return THUMBNAIL_EXTENSIONS.has(path.extname(file).toLowerCase());
}

/**
 * A small copy of a picture for the Library's cards, so scrolling a long list decodes a few dozen KB per card instead of
 * a full-size PNG each. Copies are made once and kept in `cacheDir`, named by the file's path, size and modified time (so an edited
 * picture gets a new copy). Returns null - the caller serves the original - for anything that cannot or need not be shrunk.
 */
export async function thumbnailFor(file: string, width: number, cacheDir: string, resize: Resizer): Promise<Thumbnail | null> {
  if (!canThumbnail(file)) return null;
  let stat: fs.Stats;
  try {
    stat = await fs.promises.stat(file);
  } catch {
    return null;
  }
  const key = crypto.createHash('sha1').update(`${path.resolve(file)}|${stat.size}|${stat.mtimeMs}|${width}`).digest('hex');
  const cached = path.join(cacheDir, `${key}.jpg`);
  const mime = 'image/jpeg';
  try {
    return { data: await fs.promises.readFile(cached), mime };
  } catch {
    // not made yet
  }
  let data: Buffer | null;
  try {
    data = await resize(file, width);
  } catch {
    return null;
  }
  if (!data) return null;
  try {
    await fs.promises.mkdir(cacheDir, { recursive: true });
    const partial = `${cached}.${process.pid}.part`;
    await fs.promises.writeFile(partial, data);
    await fs.promises.rename(partial, cached);
  } catch {
    // Not being able to keep it only costs making it again.
  }
  return { data, mime };
}
