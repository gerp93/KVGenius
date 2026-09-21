import * as fs from 'fs';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { getGenerationById, listFavoriteIds, setGenerationFavorite } from './db';

/** Favorited output is kept in this subfolder of the images / videos folder, so it can be picked
 * out on disk (backups, bulk edits, other tools) without going through the app. */
export const FAVORITES_FOLDER = 'favorites';

export interface OutputDirs {
  images: string;
  videos: string;
  /** Where output lived before it was split into images/videos (KVGenius_Data/images). */
  legacy: string;
}

function sameDir(a: string, b: string): boolean {
  const norm = (p: string) => (process.platform === 'win32' ? path.resolve(p).toLowerCase() : path.resolve(p));
  return norm(a) === norm(b);
}

/**
 * Where a file in `currentDir` should go when it is favorited / unfavorited, or null if it should
 * stay put: it is already in the right place, or it lives outside the app's own output folders
 * (e.g. under an old database location) where moving it would be a surprise.
 */
function targetDirFor(currentDir: string, favorite: boolean, isVideo: boolean, dirs: OutputDirs): string | null {
  const base = isVideo ? dirs.videos : dirs.images;
  const bases = [dirs.images, dirs.videos, dirs.legacy];
  if (favorite) {
    return bases.some((dir) => sameDir(dir, currentDir)) ? path.join(base, FAVORITES_FOLDER) : null;
  }
  const favoriteDirs = bases.map((dir) => path.join(dir, FAVORITES_FOLDER));
  return favoriteDirs.some((dir) => sameDir(dir, currentDir)) ? base : null;
}

/** Moves `from` into `toDir` (created if needed), picking `name-2.ext` etc. if the name is taken. */
function moveFileUnique(from: string, toDir: string): string {
  fs.mkdirSync(toDir, { recursive: true });
  const ext = path.extname(from);
  const stem = path.basename(from, ext);
  let candidate = path.join(toDir, `${stem}${ext}`);
  for (let n = 2; fs.existsSync(candidate); n++) candidate = path.join(toDir, `${stem}-${n}${ext}`);
  fs.renameSync(from, candidate);
  return candidate;
}

/**
 * Marks a generation as (un)favorited and moves its file into (or back out of) the favorites
 * folder to match. The file is moved first and the database row updated second; if the update
 * fails the file goes back, so a row never points at a file that isn't there. A file that is
 * missing or outside the output folders is left alone and only the flag changes.
 */
export function applyFavorite(
  db: DatabaseSync,
  id: number,
  favorite: boolean,
  dirs: OutputDirs,
  videoFamilies: string[]
): { imagePath: string } {
  const record = getGenerationById(db, id);
  if (!record) throw new Error('That generation no longer exists.');

  const from = path.resolve(record.imagePath);
  const isVideo = videoFamilies.includes(record.modelFamily);
  const targetDir = fs.existsSync(from) ? targetDirFor(path.dirname(from), favorite, isVideo, dirs) : null;

  let newPath = record.imagePath;
  let moved = false;
  if (targetDir) {
    newPath = moveFileUnique(from, targetDir);
    moved = true;
  }
  try {
    setGenerationFavorite(db, id, favorite, moved ? newPath : undefined);
  } catch (err) {
    if (moved) fs.renameSync(newPath, from);
    throw err;
  }
  return { imagePath: newPath };
}

/** Startup tidy-up: files of already-favorited generations (favorited before this folder existed)
 * are moved into the favorites folder. Returns how many were moved. */
export function syncFavoriteFiles(db: DatabaseSync, dirs: OutputDirs, videoFamilies: string[]): number {
  let moved = 0;
  for (const id of listFavoriteIds(db)) {
    try {
      const before = getGenerationById(db, id)?.imagePath;
      const after = applyFavorite(db, id, true, dirs, videoFamilies).imagePath;
      if (before !== after) moved++;
    } catch {
      // One unreadable/locked file should not stop the rest; it is retried on the next launch.
    }
  }
  return moved;
}
