import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GIF_FAMILY } from '../shared/gif';
import { MergeResult, MoveResult } from '../shared/workspaces';
import { FAVORITES_FOLDER, moveFileUnique } from './favorites';
import { releaseSourceImage } from './sourceImages';
import { WorkspaceDirs } from './workspaceFolders';
import { LEGACY_FOLDER_NAMES } from './workspaceFolders';
import { Recycle } from './trash';
import { appendWorkspacePromptSlots, deleteWorkspace, getWorkspace } from './workspaces';

/** What moving needs to know about where files live. */
export interface MoveContext {
  db: DatabaseSync;
  /** The `output` folder: only files inside it are moved; one anywhere else (an old database location) is left where it is. */
  outputRoot: string;
  /** The folders of a workspace; throws for one that does not exist. */
  dirsOf: (workspaceId: number) => WorkspaceDirs;
  videoFamilies: string[];
  /** Sends a file to the operating system's Recycle Bin. Needed to clear what is left in a removed workspace's folder; without it such files stay. */
  recycle?: Recycle;
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

/** Makes a workspace's folders (images, videos, gifs, sources, trash) so they exist from the moment it does, not from its first picture. */
export function ensureWorkspaceDirs(dirs: WorkspaceDirs): void {
  for (const dir of [dirs.images, dirs.videos, dirs.gifs, dirs.sources, dirs.trash]) fs.mkdirSync(dir, { recursive: true });
}

function norm(file: string): string {
  const resolved = path.resolve(file);
  return process.platform === 'win32' ? resolved.toLowerCase() : resolved;
}

function listFiles(dir: string): string[] {
  const found: string[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return found;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) found.push(...listFiles(full));
    else found.push(full);
  }
  return found;
}

/** Every file path any record or import refers to, for telling a file something needs from one nothing does. */
function referencedPaths(db: DatabaseSync): Set<string> {
  const used = new Set<string>();
  const rows = db.prepare('SELECT image_path, source_image_path, mask_image_path, trash_from, source_video_path FROM generations').all() as unknown as Record<string, string | null>[];
  for (const row of rows) for (const value of Object.values(row)) if (value) used.add(norm(value));
  for (const row of db.prepare('SELECT path FROM imports').all() as unknown as { path: string }[]) used.add(norm(row.path));
  return used;
}

/** A kept source image or mask of an item in another workspace can sit in this folder (the same picture serves several items): copy it
 * into that item's own workspace and point the item at the copy, so the folder can go. */
function rehomeKeptFiles(ctx: MoveContext, root: string): void {
  const rows = ctx.db.prepare('SELECT id, workspace_id, source_image_path, mask_image_path FROM generations WHERE source_image_path IS NOT NULL OR mask_image_path IS NOT NULL').all() as unknown as {
    id: number;
    workspace_id: number;
    source_image_path: string | null;
    mask_image_path: string | null;
  }[];
  for (const row of rows) {
    const inRoot = (file: string | null) => !!file && isInside(file, root);
    if (!inRoot(row.source_image_path) && !inRoot(row.mask_image_path)) continue;
    let sources: string;
    try {
      sources = ctx.dirsOf(row.workspace_id).sources;
    } catch {
      continue;
    }
    const source = inRoot(row.source_image_path) ? copyKept(row.source_image_path, sources) : row.source_image_path;
    const mask = inRoot(row.mask_image_path) ? copyKept(row.mask_image_path, sources) : row.mask_image_path;
    ctx.db.prepare('UPDATE generations SET source_image_path = ?, mask_image_path = ? WHERE id = ?').run(source, mask, row.id);
  }
}

/** What clearing a workspace's folder did. */
export interface FolderRemoval {
  /** The folder is gone. */
  removed: boolean;
  /** Files nothing needed that were sent to the Recycle Bin. */
  recycled: number;
  /** Files that could not be sent, or that something still needs (the folder is then kept). */
  kept: number;
}

/**
 * Removes the folder of a workspace that no longer exists (it was merged away or deleted): everything a record still needs is first
 * copied to where its own workspace keeps it, files nothing refers to any more go to the Recycle Bin (recoverable), and then the folder
 * and its sub-folders are removed, retrying where Windows has a folder briefly locked. The folder stays, with `kept` saying why, if a file
 * could not be dealt with.
 */
export async function removeWorkspaceFolder(ctx: MoveContext, root: string): Promise<FolderRemoval> {
  const result: FolderRemoval = { removed: false, recycled: 0, kept: 0 };
  if (!fs.existsSync(root)) return { ...result, removed: true };
  rehomeKeptFiles(ctx, root);
  const used = referencedPaths(ctx.db);
  for (const file of listFiles(root)) {
    if (used.has(norm(file)) || !ctx.recycle) {
      result.kept++;
      continue;
    }
    try {
      await ctx.recycle(file);
      result.recycled++;
    } catch {
      result.kept++;
    }
  }
  if (result.kept > 0) return result;
  try {
    await fs.promises.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  } catch {
    // Locked by something else: it is tried again at the next start.
  }
  result.removed = !fs.existsSync(root);
  return result;
}

const WORKSPACE_SUBFOLDERS = ['images', 'videos', 'gifs', 'sources', 'trash'];

/** Whether a folder under `output` has the shape of a workspace's folder: nothing in it but our own sub-folders (or nothing at all). */
function looksLikeWorkspaceFolder(dir: string): boolean {
  try {
    return fs.readdirSync(dir, { withFileTypes: true }).every((e) => e.isDirectory() && WORKSPACE_SUBFOLDERS.includes(e.name.toLowerCase()));
  } catch {
    return false;
  }
}

/**
 * Start-up tidy-up for folders left behind by a workspace that was merged away or deleted: a folder under `output` that belongs to no
 * workspace, is not one of the old flat folders, and has the shape of a workspace folder is cleared like any removed workspace's.
 * Anything else under `output` is left alone. Returns how many folders were removed.
 */
export async function sweepStaleWorkspaceFolders(ctx: MoveContext): Promise<number> {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(ctx.outputRoot, { withFileTypes: true });
  } catch {
    return 0;
  }
  const known = new Set(
    (ctx.db.prepare("SELECT folder FROM workspaces WHERE folder <> ''").all() as unknown as { folder: string }[]).map((r) => r.folder.toLowerCase())
  );
  let removed = 0;
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const name = entry.name.toLowerCase();
    if (known.has(name) || (LEGACY_FOLDER_NAMES as readonly string[]).includes(name)) continue;
    const dir = path.join(ctx.outputRoot, entry.name);
    if (!looksLikeWorkspaceFolder(dir)) continue;
    if ((await removeWorkspaceFolder(ctx, dir)).removed) removed++;
  }
  return removed;
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
 * the emptied workspace is deleted and its folders removed (files nothing needs any more go to the Recycle Bin). If some item could not be moved the workspace is kept, with what is left in it.
 */
export async function mergeWorkspaces(ctx: MoveContext, fromId: number, intoId: number): Promise<MergeResult> {
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
    // Its Generate tabs are added after the other workspace's, so nothing typed there is lost.
    appendWorkspacePromptSlots(db, fromId, intoId);
    deleteWorkspace(db, fromId);
    await removeWorkspaceFolder(ctx, fromRoot);
    deleted = true;
  }
  return { ...moved, deleted, renamedStyles };
}
