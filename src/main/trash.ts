import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { TrashMoveResult, TrashStats, cutoffIso } from '../shared/cleanup';
import { GenerationRecord } from '../shared/types';
import { getGenerationById } from './db';
import { moveFileUnique } from './favorites';
import { releaseSourceImage } from './sourceImages';

/**
 * The Trash. Moving an item here takes it out of the Library but keeps its file (in the trash folder)
 * and its record, so it can be put back; only deleting from the Trash is for good. Favorites and
 * pinned items are never moved here, whoever asks: that is what "kept" means.
 */

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

/** Moves the given generations into the Trash. Each file is moved first and its record updated second;
 * if the update fails the file goes back, so a record never points at a file that is not there. A file
 * that is already missing is still trashed (there is just nothing to move). */
export function moveToTrash(db: DatabaseSync, ids: number[], trashDir: string, now: Date = new Date()): TrashMoveResult {
  const result: TrashMoveResult = { moved: 0, skipped: 0, failed: 0 };
  const find = db.prepare('SELECT id, image_path, favorite, pinned_at, trashed_at FROM generations WHERE id = ?');
  const mark = db.prepare('UPDATE generations SET trashed_at = ?, trash_from = ?, image_path = ? WHERE id = ?');
  for (const id of ids) {
    const row = find.get(id) as unknown as
      | { id: number; image_path: string; favorite: number; pinned_at: string | null; trashed_at: string | null }
      | undefined;
    if (!row) continue;
    if (row.trashed_at !== null || row.favorite === 1 || row.pinned_at !== null) {
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

/** Cleanup: moves everything `cleanupCandidates` finds. */
export function runCleanup(db: DatabaseSync, days: number, trashDir: string, now: Date = new Date()): TrashMoveResult {
  return moveToTrash(db, cleanupCandidates(db, days, now).map((c) => c.id), trashDir, now);
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

function deleteRows(db: DatabaseSync, rows: Row[], sourcesDir: string): number {
  const sources = db.prepare('SELECT source_image_path FROM generations WHERE id = ?');
  const remove = db.prepare('DELETE FROM generations WHERE id = ?');
  let deleted = 0;
  for (const row of rows) {
    const source = (sources.get(row.id) as unknown as { source_image_path: string | null } | undefined)?.source_image_path ?? null;
    remove.run(row.id);
    try {
      fs.unlinkSync(row.image_path);
    } catch {
      // Already gone - the record is deleted either way.
    }
    releaseSourceImage(db, source, sourcesDir);
    deleted++;
  }
  return deleted;
}

/** Deletes items from the Trash for good, files included. Only ever touches items that are in the Trash. */
export function deleteFromTrash(db: DatabaseSync, ids: number[], sourcesDir: string): number {
  const find = db.prepare('SELECT id, image_path, trash_from FROM generations WHERE id = ? AND trashed_at IS NOT NULL');
  const rows = ids.map((id) => find.get(id) as unknown as Row | undefined).filter((r): r is Row => r !== undefined);
  return deleteRows(db, rows, sourcesDir);
}

/** Empties the Trash. With `olderThanDays`, only what has been there at least that long. */
export function emptyTrash(db: DatabaseSync, sourcesDir: string, olderThanDays: number | null = null, now: Date = new Date()): number {
  const rows = (
    olderThanDays === null
      ? db.prepare('SELECT id, image_path, trash_from FROM generations WHERE trashed_at IS NOT NULL').all()
      : db
          .prepare('SELECT id, image_path, trash_from FROM generations WHERE trashed_at IS NOT NULL AND trashed_at < ?')
          .all(cutoffIso(olderThanDays, now))
  ) as unknown as Row[];
  return deleteRows(db, rows, sourcesDir);
}
