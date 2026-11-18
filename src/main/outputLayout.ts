import * as fs from 'fs';
import * as path from 'path';
import { DEFAULT_WORKSPACE_ID, workspaceFolder } from './workspaces';
import { LEGACY_FOLDER_NAMES, workspaceDirs } from './workspaceFolders';
import { MoveContext, moveGenerations } from './moveItems';

/** What the move into per-workspace folders did on this start. */
export interface LayoutResult {
  /** Folders (images, videos, ...) that were moved under the first workspace's folder. */
  foldersMoved: number;
  /** Records whose file paths were rewritten to follow them. */
  rowsRewritten: number;
  /** Items of other workspaces moved into their own workspace's folder. */
  itemsRelocated: number;
  /** Things that could not be moved now (a locked file...); they are tried again next start. */
  problems: number;
  /** The copy of the database made before anything was changed (never deleted by the app), or null when nothing needed changing. */
  backup: string | null;
}

/** A backup name that does not exist yet (VACUUM INTO refuses to overwrite). The name pattern is what Settings lists as a migration backup. */
function freeBackupPath(dbPath: string): string {
  const base = `${dbPath}.pre-workspace-folders`;
  if (!fs.existsSync(base)) return base;
  for (let n = 2; ; n++) if (!fs.existsSync(`${base}-${n}`)) return `${base}-${n}`;
}

/** Moves a folder to a new place: the whole folder at once when the new place is free (a rename on the same drive: instant, all or
 * nothing), else its contents one by one without ever overwriting anything. Returns how many things could not be moved. */
function adoptDir(from: string, to: string): { moved: boolean; problems: number } {
  let problems = 0;
  try {
    // An empty folder already there (made ahead of time) is no reason to move file by file.
    if (fs.existsSync(to) && fs.readdirSync(to).length === 0) fs.rmdirSync(to);
    if (!fs.existsSync(to)) {
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.renameSync(from, to);
      return { moved: true, problems };
    }
    for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
      const source = path.join(from, entry.name);
      const destination = path.join(to, entry.name);
      if (entry.isDirectory()) {
        problems += adoptDir(source, destination).problems;
      } else if (!fs.existsSync(destination)) {
        fs.renameSync(source, destination);
      } else {
        problems++;
      }
    }
    try {
      fs.rmdirSync(from);
    } catch {
      // Something is left in it (a name clash): leave the folder.
    }
    return { moved: true, problems };
  } catch {
    return { moved: false, problems: problems + 1 };
  }
}

/** A stored path rewritten from the old flat layout (`output/images/x.png`) into a workspace's folder, or null when it is not in the flat layout. */
export function followLegacyPath(file: string | null, outputRoot: string, folder: string): string | null {
  if (!file) return null;
  const rel = path.relative(path.resolve(outputRoot), path.resolve(file));
  if (rel === '' || rel.startsWith('..') || path.isAbsolute(rel)) return null;
  const first = rel.split(path.sep)[0].toLowerCase();
  if (!(LEGACY_FOLDER_NAMES as readonly string[]).includes(first)) return null;
  return path.join(outputRoot, folder, rel);
}

const PATH_COLUMNS = ['image_path', 'source_image_path', 'mask_image_path', 'trash_from', 'source_video_path'] as const;

/**
 * One-time (and safe to repeat) move from the flat layout, where every workspace's files shared `output/{images,videos,gifs,sources,trash}`,
 * to a folder per workspace:
 *  1. the flat folders move whole under the first workspace's folder (`output/Default/images`, ...);
 *  2. every stored path that pointed into them is rewritten to follow;
 *  3. items that belong to another workspace (a Hidden workspace, say) move on into their own workspace's folder.
 * Each step can be interrupted: the next start finds what is left and carries on, because the folders and the paths are both checked every time.
 */
export function migrateOutputLayout(ctx: MoveContext, dbPath: string | null = null): LayoutResult {
  const { db, outputRoot } = ctx;
  const result: LayoutResult = { foldersMoved: 0, rowsRewritten: 0, itemsRelocated: 0, problems: 0, backup: null };
  const firstFolder = workspaceFolder(db, DEFAULT_WORKSPACE_ID);
  if (!firstFolder) return result;
  const first = workspaceDirs(outputRoot, firstFolder);

  const isDir = (dir: string) => {
    try {
      return fs.statSync(dir).isDirectory();
    } catch {
      return false; // Not there (a new install, or already moved).
    }
  };
  // Before the first real change, the database is copied next to itself (VACUUM INTO, so what is still in the write-ahead log is included).
  // If the copy cannot be made this throws and nothing is changed: the old layout keeps working and the next start tries again.
  if (dbPath && dbPath !== ':memory:' && LEGACY_FOLDER_NAMES.some((name) => isDir(path.join(outputRoot, name)))) {
    result.backup = freeBackupPath(dbPath);
    db.exec(`VACUUM INTO '${result.backup.replace(/'/g, "''")}'`);
  }

  for (const name of LEGACY_FOLDER_NAMES) {
    const from = path.join(outputRoot, name);
    if (!isDir(from)) continue;
    const moved = adoptDir(from, path.join(first.root, name));
    if (moved.moved) result.foldersMoved++;
    result.problems += moved.problems;
  }

  const rows = db.prepare(`SELECT id, ${PATH_COLUMNS.join(', ')} FROM generations`).all() as unknown as Record<string, string | number | null>[];
  const update = (column: string) => db.prepare(`UPDATE generations SET ${column} = ? WHERE id = ?`);
  const setters = Object.fromEntries(PATH_COLUMNS.map((c) => [c, update(c)]));
  db.exec('BEGIN');
  try {
    for (const row of rows) {
      let changed = false;
      for (const column of PATH_COLUMNS) {
        const value = row[column] as string | null;
        const next = followLegacyPath(value, outputRoot, firstFolder);
        if (next && next !== value) {
          setters[column].run(next, row.id as number);
          changed = true;
        }
      }
      if (changed) result.rowsRewritten++;
    }
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }

  // Items of other workspaces are now in the first workspace's folders: move each into its own.
  const others = db.prepare('SELECT id, workspace_id, image_path FROM generations WHERE workspace_id <> ?').all(DEFAULT_WORKSPACE_ID) as unknown as {
    id: number;
    workspace_id: number;
    image_path: string;
  }[];
  const misplaced = new Map<number, number[]>();
  for (const item of others) {
    let dirs;
    try {
      dirs = ctx.dirsOf(item.workspace_id);
    } catch {
      continue;
    }
    const rel = path.relative(path.resolve(dirs.root), path.resolve(item.image_path));
    const insideOwn = rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
    const insideOutput = path.relative(path.resolve(outputRoot), path.resolve(item.image_path)).startsWith('..') === false;
    if (!insideOwn && insideOutput) misplaced.set(item.workspace_id, [...(misplaced.get(item.workspace_id) ?? []), item.id]);
  }
  for (const [workspaceId, ids] of misplaced) {
    const moved = moveGenerations(ctx, ids, workspaceId);
    result.itemsRelocated += moved.moved;
    result.problems += moved.failed;
  }
  return result;
}
