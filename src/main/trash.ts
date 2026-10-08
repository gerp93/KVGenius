import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { TrashEmptyResult, TrashMoveResult, TrashStats, cutoffIso } from '../shared/cleanup';
import { GenerationRecord } from '../shared/types';
import { getGenerationById } from './db';
import { moveFileUnique } from './favorites';
import { releaseSourceImage } from './sourceImages';

/**
 * The Trash - step one of a two-step delete. Moving an item here takes it out of the Library but keeps
 * its file (in the trash folder) and its record, so it can be put back. Step two, emptying the Trash,
 * hands the files to the operating system's Recycle Bin (`recycle`), so even that is recoverable
 * outside the app.
 *
 * The automatic cleanup never trashes a favorite or a pinned item; an explicit delete by the user can
 * (`includeKept`).
 */

/** Sends a file to the operating system's Recycle Bin (Electron's `shell.trashItem`). Rejects if it cannot. */
export type Recycle = (file: string) => Promise<void>;

interface Row {
  id: number;
  image_path: string;
  trash_from: string | null;
}

function totalBytes(paths: string[]): number {
  let bytes = 0;
  for (const file of paths) {
    try {
      bytes += fs.statSync(file).size;
    } catch {
      // Already gone: it takes no space.
    }
  }
  return bytes;
}

/** What a cleanup of items older than `days` days would move: not favorited, not pinned, not already trashed. */
export function cleanupCandidates(db: DatabaseSync, days: number, now: Date = new Date()): { id: number; imagePath: string }[] {
  const rows = db
    .prepare(
      `SELECT id, image_path FROM generations
       WHERE trashed_at IS NULL AND favorite = 0 AND pinned_at IS NULL AND created_at < ? ORDER BY id`
    )
    .all(cutoffIso(days, now)) as unknown as { id: number; image_path: string }[];
  return rows.map((r) => ({ id: r.id, imagePath: r.image_path }));
}

export function previewCleanup(db: DatabaseSync, days: number, now: Date = new Date()): TrashStats {
  const candidates = cleanupCandidates(db, days, now);
  return { count: candidates.length, bytes: totalBytes(candidates.map((c) => c.imagePath)) };
}

export interface MoveOptions {
  now?: Date;
  /** Also move favorites and pinned items. Only for something the user asked for item by item (the
   * Delete button); the automatic cleanup never sets it, and neither does "trash the rest" after a compare. */
  includeKept?: boolean;
}

/** Moves the given generations into the Trash. Each file is moved first and its record updated second;
 * if the update fails the file goes back, so a record never points at a file that is not there. A file
 * that is already missing is still trashed (there is just nothing to move). */
export function moveToTrash(db: DatabaseSync, ids: number[], trashDir: string, options: MoveOptions = {}): TrashMoveResult {
  const now = options.now ?? new Date();
  const result: TrashMoveResult = { moved: 0, skipped: 0, failed: 0 };
  const find = db.prepare('SELECT id, image_path, favorite, pinned_at, trashed_at FROM generations WHERE id = ?');
  const mark = db.prepare('UPDATE generations SET trashed_at = ?, trash_from = ?, image_path = ? WHERE id = ?');
  for (const id of ids) {
    const row = find.get(id) as unknown as
      | { id: number; image_path: string; favorite: number; pinned_at: string | null; trashed_at: string | null }
      | undefined;
    if (!row) continue;
    const kept = row.favorite === 1 || row.pinned_at !== null;
    if (row.trashed_at !== null || (kept && !options.includeKept)) {
      result.skipped++;
      continue;
    }
    const from = path.resolve(row.image_path);
    let newPath = row.image_path;
    let movedFile = false;
    try {
      if (fs.existsSync(from)) {
        newPath = moveFileUnique(from, trashDir);
        movedFile = true;
      }
    } catch {
      result.failed++;
      continue;
    }
    try {
      mark.run(now.toISOString(), row.image_path, newPath, id);
      result.moved++;
    } catch {
      if (movedFile) fs.renameSync(newPath, from);
      result.failed++;
    }
  }
  return result;
}

/** Cleanup: moves everything `cleanupCandidates` finds. Never a favorite or pinned item. */
export function runCleanup(db: DatabaseSync, days: number, trashDir: string, now: Date = new Date()): TrashMoveResult {
  return moveToTrash(db, cleanupCandidates(db, days, now).map((c) => c.id), trashDir, { now });
}

export function trashStats(db: DatabaseSync): TrashStats {
  const rows = db.prepare('SELECT image_path FROM generations WHERE trashed_at IS NOT NULL').all() as unknown as { image_path: string }[];
  return { count: rows.length, bytes: totalBytes(rows.map((r) => r.image_path)) };
}

/** Everything in the Trash, newest generation first. Paged by id like the Library. */
export function listTrashed(db: DatabaseSync, limit: number, beforeId: number | null): GenerationRecord[] {
  const ids = db
    .prepare(`SELECT id FROM generations WHERE trashed_at IS NOT NULL ${beforeId === null ? '' : 'AND id < ?'} ORDER BY id DESC LIMIT ?`)
    .all(...(beforeId === null ? [limit] : [beforeId, limit])) as unknown as { id: number }[];
  return ids.map((r) => getGenerationById(db, r.id)).filter((r): r is GenerationRecord => r !== null);
}

/** Puts items back where they came from (under a new name if that name is now taken). Items whose file
 * is gone cannot be restored and are left in the Trash. */
export function restoreFromTrash(db: DatabaseSync, ids: number[]): { restored: number; failed: number } {
  let restored = 0;
  let failed = 0;
  const find = db.prepare('SELECT id, image_path, trash_from FROM generations WHERE id = ? AND trashed_at IS NOT NULL');
  const unmark = db.prepare('UPDATE generations SET trashed_at = NULL, trash_from = NULL, image_path = ? WHERE id = ?');
  for (const id of ids) {
    const row = find.get(id) as unknown as Row | undefined;
    if (!row) continue;
    const current = path.resolve(row.image_path);
    if (!fs.existsSync(current)) {
      failed++;
      continue;
    }
    // Back into the folder it came from; if it was never moved (it was already missing), it stays put.
    const originalDir = path.dirname(row.trash_from ?? row.image_path);
    let restoredPath = current;
    let movedFile = false;
    try {
      if (path.resolve(originalDir) !== path.dirname(current)) {
        restoredPath = moveFileUnique(current, originalDir);
        movedFile = true;
      }
    } catch {
      failed++;
      continue;
    }
    try {
      unmark.run(restoredPath, id);
      restored++;
    } catch {
      if (movedFile) fs.renameSync(restoredPath, current);
      failed++;
    }
  }
  return { restored, failed };
}

/** Sends each item's file to the Recycle Bin, then deletes its record. An item the Recycle Bin will not
 * take stays in the Trash, whole, and is counted as failed. */
async function emptyRows(db: DatabaseSync, rows: Row[], sourcesDir: string, recycle: Recycle): Promise<TrashEmptyResult> {
  const sources = db.prepare('SELECT source_image_path, mask_image_path FROM generations WHERE id = ?');
  const remove = db.prepare('DELETE FROM generations WHERE id = ?');
  const result: TrashEmptyResult = { deleted: 0, failed: 0 };
  for (const row of rows) {
    if (fs.existsSync(row.image_path)) {
      try {
        await recycle(row.image_path);
      } catch {
        result.failed++;
        continue;
      }
    }
    const kept = sources.get(row.id) as unknown as { source_image_path: string | null; mask_image_path: string | null } | undefined;
    remove.run(row.id);
    // The kept copy of a video's source image (and an inpainting result's mask) is the app's own duplicate, not something to recycle.
    releaseSourceImage(db, kept?.source_image_path ?? null, sourcesDir);
    releaseSourceImage(db, kept?.mask_image_path ?? null, sourcesDir);
    result.deleted++;
  }
  return result;
}

/** Removes the given items from the Trash: their files go to the Recycle Bin and they can no longer be
 * restored into the app. Only ever touches items that are in the Trash. */
export async function deleteFromTrash(db: DatabaseSync, ids: number[], sourcesDir: string, recycle: Recycle): Promise<TrashEmptyResult> {
  const find = db.prepare('SELECT id, image_path, trash_from FROM generations WHERE id = ? AND trashed_at IS NOT NULL');
  const rows = ids.map((id) => find.get(id) as unknown as Row | undefined).filter((r): r is Row => r !== undefined);
  return emptyRows(db, rows, sourcesDir, recycle);
}

/** Empties the Trash into the Recycle Bin. With `olderThanDays`, only what has been there at least that long. */
export async function emptyTrash(
  db: DatabaseSync,
  sourcesDir: string,
  recycle: Recycle,
  olderThanDays: number | null = null,
  now: Date = new Date()
): Promise<TrashEmptyResult> {
  const rows = (
    olderThanDays === null
      ? db.prepare('SELECT id, image_path, trash_from FROM generations WHERE trashed_at IS NOT NULL').all()
      : db
          .prepare('SELECT id, image_path, trash_from FROM generations WHERE trashed_at IS NOT NULL AND trashed_at < ?')
          .all(cutoffIso(olderThanDays, now))
  ) as unknown as Row[];
  return emptyRows(db, rows, sourcesDir, recycle);
}
