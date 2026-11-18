import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GIF_FAMILY } from '../shared/gif';
import { MergeResult, MoveResult } from '../shared/workspaces';
import { FAVORITES_FOLDER, moveFileUnique } from './favorites';
import { releaseSourceImage } from './sourceImages';
import { WorkspaceDirs } from './workspaceFolders';
import { deleteWorkspace, getWorkspace } from './workspaces';

/** What moving needs to know about where files live. */
export interface MoveContext {
  db: DatabaseSync;
  /** The `output` folder: only files inside it are moved; one anywhere else (an old database location) is left where it is. */
  outputRoot: string;
  /** The folders of a workspace; throws for one that does not exist. */
  dirsOf: (workspaceId: number) => WorkspaceDirs;
  videoFamilies: string[];
}

interface Row {
  id: number;
  workspace_id: number;
  model_family: string;
  image_path: string;
  favorite: number;
  trashed_at: string | null;
  trash_from: string | null;
  source_image_path: string | null;
  mask_image_path: string | null;
}

function isInside(file: string, dir: string): boolean {
  const rel = path.relative(path.resolve(dir), path.resolve(file));
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

/** Copies a kept source image or mask into a workspace's sources folder (they are named by their contents, so the same name is the same
 * picture) and returns the copy's path - or the original path when there is nothing to copy (it is gone, or already there). */
function copyKept(file: string | null, sourcesDir: string): string | null {
  if (!file) return null;
  const target = path.join(sourcesDir, path.basename(file));
  if (path.resolve(target) === path.resolve(file)) return file;
  try {
    if (!fs.existsSync(file)) return file;
    fs.mkdirSync(sourcesDir, { recursive: true });
    if (!fs.existsSync(target)) fs.copyFileSync(file, target);
    return target;
  } catch {
    return file;
  }
}

/**
 * Puts items into a workspace: their workspace changes and their files move into that workspace's folders (images, videos or gifs, in its
 * favorites folder for a favorite; the Trash folder for an item in the Trash). The kept source images and masks they were made from are
 * copied (the same picture may serve items elsewhere) and the old copy goes once nothing uses it. Each file is moved first and its record
 * updated second, and moved back if the update fails, so a record never points at a file that is not there. Items already in that
 * workspace are only tidied into its folders; one whose file is missing, or outside the output folder, changes workspace but its file stays.
 */
export function moveGenerations(ctx: MoveContext, ids: number[], targetId: number): MoveResult {
  const result: MoveResult = { moved: 0, skipped: 0, failed: 0 };
  const { db } = ctx;
  const target = ctx.dirsOf(targetId);
  const find = db.prepare(
    'SELECT id, workspace_id, model_family, image_path, favorite, trashed_at, trash_from, source_image_path, mask_image_path FROM generations WHERE id = ?'
  );
  const update = db.prepare(
    'UPDATE generations SET workspace_id = ?, image_path = ?, trash_from = ?, source_image_path = ?, mask_image_path = ? WHERE id = ?'
  );
  const repoint = db.prepare('UPDATE generations SET source_video_path = ? WHERE source_video_path = ?');

  for (const id of new Set(ids)) {
    const row = find.get(id) as unknown as Row | undefined;
    if (!row) {
      result.skipped++;
      continue;
    }
    const base = ctx.videoFamilies.includes(row.model_family) ? target.videos : row.model_family === GIF_FAMILY ? target.gifs : target.images;
    const trashed = row.trashed_at !== null;
    const libraryDir = row.favorite === 1 ? path.join(base, FAVORITES_FOLDER) : base;
    const targetDir = trashed ? target.trash : libraryDir;

    const from = path.resolve(row.image_path);
    const movable = fs.existsSync(from) && isInside(from, ctx.outputRoot);
    const wrongPlace = movable && path.resolve(path.dirname(from)) !== path.resolve(targetDir);
    const needsKept = (file: string | null) =>
      !!file && isInside(file, ctx.outputRoot) && path.resolve(path.join(target.sources, path.basename(file))) !== path.resolve(file);
    if (row.workspace_id === targetId && !wrongPlace && !needsKept(row.source_image_path) && !needsKept(row.mask_image_path)) {
      result.skipped++;
      continue;
    }

    let newPath = row.image_path;
    let movedFile = false;
    try {
      if (wrongPlace) {
        newPath = moveFileUnique(from, targetDir);
        movedFile = true;
      }
    } catch {
      result.failed++;
      continue;
    }
    // Where it would go back to if it leaves the Trash: the library folder, under the file's name.
    const newTrashFrom = trashed ? path.join(libraryDir, path.basename(row.trash_from ?? row.image_path)) : row.trash_from;
    const keptIn = (file: string | null) => (file && isInside(file, ctx.outputRoot) ? copyKept(file, target.sources) : file);
    const newSource = keptIn(row.source_image_path);
    const newMask = keptIn(row.mask_image_path);
    try {
      db.exec('BEGIN');
      update.run(targetId, newPath, newTrashFrom, newSource, newMask, id);
      // A video-to-video result reads its source video in place: anything that pointed at this file follows it.
      if (movedFile) repoint.run(newPath, row.image_path);
      db.exec('COMMIT');
    } catch {
      try {
        db.exec('ROLLBACK');
      } catch {
        // No transaction to roll back.
      }
      if (movedFile) fs.renameSync(newPath, from);
      result.failed++;
      continue;
    }
    // The old copies of the source and mask go once nothing uses them.
    for (const [before, after] of [[row.source_image_path, newSource], [row.mask_image_path, newMask]]) {
      if (before && after && path.resolve(before) !== path.resolve(after)) releaseSourceImage(db, before, path.dirname(before));
    }
    result.moved++;
  }
  return result;
}

/** Removes a folder and any folders inside it that are empty; whatever has a file in it stays. */
export function pruneEmptyDirs(dir: string): void {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) if (entry.isDirectory()) pruneEmptyDirs(path.join(dir, entry.name));
  try {
    fs.rmdirSync(dir);
  } catch {
    // Not empty (or in use): leave it.
  }
}

function freeStyleName(db: DatabaseSync, name: string, suffix: string, intoId: number): string {
  const taken = db.prepare('SELECT 1 FROM styles WHERE name = ? COLLATE NOCASE AND (workspace_id IS NULL OR workspace_id = ?)');
  for (let n = 1; ; n++) {
    const candidate = `${name.slice(0, 40)} (${suffix}${n > 1 ? ` ${n}` : ''})`;
    if (!taken.get(candidate, intoId)) return candidate;
  }
}

/**
 * Merges one workspace into another: every item (the Trash too) moves across with its files, its imports, assemblies and queue history
 * follow, and its own styles and elements become the other's (renamed "Name (from X)" if the other already has one with that name). Then
 * the emptied workspace is deleted and its folders removed. If some item could not be moved the workspace is kept, with what is left in it.
 */
export function mergeWorkspaces(ctx: MoveContext, fromId: number, intoId: number): MergeResult {
  const { db } = ctx;
  const from = getWorkspace(db, fromId);
  const into = getWorkspace(db, intoId);
  if (!from || !into) throw new Error('That workspace no longer exists.');
  if (fromId === intoId) throw new Error('Choose a different workspace to merge into.');
  const fromRoot = ctx.dirsOf(fromId).root;

  const ids = (db.prepare('SELECT id FROM generations WHERE workspace_id = ? ORDER BY id').all(fromId) as unknown as { id: number }[]).map((r) => r.id);
  const moved = moveGenerations(ctx, ids, intoId);

  for (const table of ['imports', 'assemblies', 'jobs']) db.prepare(`UPDATE ${table} SET workspace_id = ? WHERE workspace_id = ?`).run(intoId, fromId);

  let renamedStyles = 0;
  const scoped = db.prepare('SELECT id, name FROM styles WHERE workspace_id = ?').all(fromId) as unknown as { id: number; name: string }[];
  const clash = db.prepare('SELECT 1 FROM styles WHERE name = ? COLLATE NOCASE AND id <> ? AND (workspace_id IS NULL OR workspace_id = ?)');
  for (const style of scoped) {
    let name = style.name;
    if (clash.get(name, style.id, intoId)) {
      name = freeStyleName(db, name, `from ${from.name}`, intoId);
      renamedStyles++;
    }
    db.prepare('UPDATE styles SET workspace_id = ?, name = ? WHERE id = ?').run(intoId, name, style.id);
  }

  const left = db.prepare('SELECT COUNT(*) AS n FROM generations WHERE workspace_id = ?').get(fromId) as unknown as { n: number };
  let deleted = false;
  if (left.n === 0) {
    deleteWorkspace(db, fromId);
    pruneEmptyDirs(fromRoot);
    deleted = true;
  }
  return { ...moved, deleted, renamedStyles };
}
