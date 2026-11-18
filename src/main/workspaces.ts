import { DatabaseSync } from 'node:sqlite';
import { CleanupSettings, DEFAULT_CLEANUP_SETTINGS, normalizeCleanupSettings } from '../shared/cleanup';
import { WorkspaceDirs, folderNameFor, workspaceDirs } from './workspaceFolders';
import {
  DEFAULT_WORKSPACE_NAME,
  HIDDEN_WORKSPACE_NAME,
  STYLE_EXAMPLES_WORKSPACE_NAME,
  Workspace,
  WorkspaceSummary,
  WorkspaceUpdate,
  validateWorkspaceName,
} from '../shared/workspaces';

/**
 * Workspaces (see shared/workspaces.ts). `system` marks the one workspace the app keeps for itself (the Styles
 * page's example pictures); it is never listed, never opened and cannot be changed from the UI.
 */
export const WORKSPACES_SCHEMA = `
CREATE TABLE IF NOT EXISTS workspaces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  theme TEXT,
  hidden INTEGER NOT NULL DEFAULT 0,
  system INTEGER NOT NULL DEFAULT 0,
  cleanup TEXT,
  folder TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
`;

/** A folder name for a workspace about to be made: from its name, different from every other workspace's folder. */
function nextFolderName(db: DatabaseSync, name: string): string {
  const taken = (db.prepare("SELECT folder FROM workspaces WHERE folder <> ''").all() as unknown as { folder: string }[]).map((r) => r.folder);
  return folderNameFor(name, taken);
}

/** Gives a folder to every workspace made before workspaces had folders of their own. Safe to run on every start. */
export function assignWorkspaceFolders(db: DatabaseSync): void {
  const rows = db.prepare("SELECT id, name FROM workspaces WHERE folder = '' ORDER BY id").all() as unknown as { id: number; name: string }[];
  for (const row of rows) db.prepare('UPDATE workspaces SET folder = ? WHERE id = ?').run(nextFolderName(db, row.name), row.id);
}

/** The folders of a workspace (any, including the app's own). Throws for one that does not exist. */
export function workspaceDirsOf(db: DatabaseSync, outputRoot: string, id: number): WorkspaceDirs {
  const folder = workspaceFolder(db, id);
  if (!folder) throw new Error('That workspace no longer exists.');
  return workspaceDirs(outputRoot, folder);
}

/** The folder name of a workspace (any, including the app's own), or null if there is no such workspace. */
export function workspaceFolder(db: DatabaseSync, id: number): string | null {
  const row = db.prepare("SELECT folder FROM workspaces WHERE id = ? AND folder <> ''").get(id) as unknown as { folder: string } | undefined;
  return row?.folder ?? null;
}

/** The workspace everything made before workspaces existed belongs to. Its id is fixed, because the
 * `workspace_id` columns default to it. */
export const DEFAULT_WORKSPACE_ID = 1;

interface WorkspaceRow {
  id: number;
  name: string;
  theme: string | null;
  hidden: number;
  system: number;
  cleanup: string | null;
  created_at: string;
}

function parseCleanup(json: string | null): CleanupSettings {
  if (!json) return { ...DEFAULT_CLEANUP_SETTINGS };
  try {
    return normalizeCleanupSettings(JSON.parse(json));
  } catch {
    return { ...DEFAULT_CLEANUP_SETTINGS };
  }
}

function rowToWorkspace(row: WorkspaceRow): Workspace {
  return {
    id: row.id,
    name: row.name,
    theme: row.theme && row.theme !== '' ? row.theme : null,
    hidden: row.hidden === 1,
    cleanup: parseCleanup(row.cleanup),
    createdAt: row.created_at,
  };
}

function isNameTaken(db: DatabaseSync, name: string, exceptId: number | null): boolean {
  const row = db
    .prepare('SELECT id FROM workspaces WHERE system = 0 AND name = ? COLLATE NOCASE AND id IS NOT ?')
    .get(name, exceptId) as unknown as { id: number } | undefined;
  return row !== undefined;
}

/** The workspaces you can open, alphabetical (ignoring case); hidden ones only when asked for. */
export function listWorkspaces(db: DatabaseSync, includeHidden: boolean): Workspace[] {
  const rows = db
    .prepare(`SELECT * FROM workspaces WHERE system = 0 ${includeHidden ? '' : 'AND hidden = 0'} ORDER BY name COLLATE NOCASE, id`)
    .all() as unknown as WorkspaceRow[];
  return rows.map(rowToWorkspace);
}

/** One workspace you can open (never the app's own), or null. */
export function getWorkspace(db: DatabaseSync, id: number): Workspace | null {
  const row = db.prepare('SELECT * FROM workspaces WHERE id = ? AND system = 0').get(id) as unknown as WorkspaceRow | undefined;
  return row ? rowToWorkspace(row) : null;
}

/** The workspace used when none has been chosen yet: the oldest one you can open. */
export function defaultWorkspaceId(db: DatabaseSync): number {
  const row = db.prepare('SELECT id FROM workspaces WHERE system = 0 ORDER BY hidden, id LIMIT 1').get() as unknown as { id: number } | undefined;
  return row?.id ?? DEFAULT_WORKSPACE_ID;
}

let activeWorkspaceId: number | null = null;

/** The window tells the main process which workspace it is in; null while it shows the grid. */
export function setActiveWorkspaceId(id: number | null): void {
  activeWorkspaceId = id;
}

/** The workspace the window is in - or, before one is chosen (or if it was deleted), the default one.
 * Everything the app does for "the current workspace" (a Library listing, a new job, an outside client's
 * request that names none) goes through here. */
export function currentWorkspaceId(db: DatabaseSync): number {
  if (activeWorkspaceId !== null && getWorkspace(db, activeWorkspaceId)) return activeWorkspaceId;
  return defaultWorkspaceId(db);
}

export function createWorkspace(db: DatabaseSync, rawName: unknown, now: Date = new Date()): Workspace {
  const checked = validateWorkspaceName(rawName);
  if (!checked.ok) throw new Error(checked.message);
  if (isNameTaken(db, checked.name, null)) throw new Error(`A workspace named "${checked.name}" already exists.`);
  const result = db
    .prepare('INSERT INTO workspaces (name, theme, hidden, system, cleanup, folder, created_at) VALUES (?, NULL, 0, 0, ?, ?, ?)')
    .run(checked.name, JSON.stringify(DEFAULT_CLEANUP_SETTINGS), nextFolderName(db, checked.name), now.toISOString());
  return getWorkspace(db, Number(result.lastInsertRowid)) as Workspace;
}

export function updateWorkspace(db: DatabaseSync, id: number, patch: WorkspaceUpdate): Workspace {
  const current = getWorkspace(db, id);
  if (!current) throw new Error('That workspace no longer exists.');
  let name = current.name;
  if (patch.name !== undefined) {
    const checked = validateWorkspaceName(patch.name);
    if (!checked.ok) throw new Error(checked.message);
    if (isNameTaken(db, checked.name, id)) throw new Error(`A workspace named "${checked.name}" already exists.`);
    name = checked.name;
  }
  const theme = patch.theme === undefined ? current.theme : typeof patch.theme === 'string' && patch.theme.trim() !== '' ? patch.theme.trim() : null;
  const hidden = patch.hidden === undefined ? current.hidden : patch.hidden === true;
  db.prepare('UPDATE workspaces SET name = ?, theme = ?, hidden = ? WHERE id = ?').run(name, theme, hidden ? 1 : 0, id);
  return getWorkspace(db, id) as Workspace;
}

/** Saves a workspace's retention rules. */
export function saveWorkspaceCleanup(db: DatabaseSync, id: number, settings: CleanupSettings): CleanupSettings {
  const cleaned = normalizeCleanupSettings(settings);
  if (!getWorkspace(db, id)) throw new Error('That workspace no longer exists.');
  db.prepare('UPDATE workspaces SET cleanup = ? WHERE id = ?').run(JSON.stringify(cleaned), id);
  return cleaned;
}

/** How many things a workspace holds (the Library and Trash, plus imports and assembled videos). */
function holdsAnything(db: DatabaseSync, id: number): boolean {
  for (const table of ['generations', 'imports', 'assemblies', 'jobs']) {
    const row = db.prepare(`SELECT COUNT(*) AS n FROM ${table} WHERE workspace_id = ?`).get(id) as unknown as { n: number };
    if (row.n > 0) return true;
  }
  return false;
}

/** Deletes a workspace that holds nothing (there is no way to delete one with things in it - move or trash
 * them first). The last workspace you can open stays. Styles that were only for this workspace go with it. */
export function deleteWorkspace(db: DatabaseSync, id: number): void {
  if (!getWorkspace(db, id)) return;
  if (listWorkspaces(db, true).length <= 1) throw new Error('That is the only workspace, so it stays.');
  if (holdsAnything(db, id)) throw new Error('A workspace can only be deleted when it is empty. Empty its Trash and delete what is left first.');
  db.exec('BEGIN');
  try {
    db.prepare('DELETE FROM styles WHERE workspace_id = ?').run(id);
    db.prepare('DELETE FROM workspaces WHERE id = ?').run(id);
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}

/** The workspace the Styles page's example pictures live in, made on first use. */
export function styleExamplesWorkspaceId(db: DatabaseSync, now: Date = new Date()): number {
  const found = db.prepare('SELECT id FROM workspaces WHERE system = 1 LIMIT 1').get() as unknown as { id: number } | undefined;
  if (found) return found.id;
  const result = db
    .prepare('INSERT INTO workspaces (name, theme, hidden, system, cleanup, folder, created_at) VALUES (?, NULL, 1, 1, ?, ?, ?)')
    .run(STYLE_EXAMPLES_WORKSPACE_NAME, JSON.stringify(DEFAULT_CLEANUP_SETTINGS), nextFolderName(db, STYLE_EXAMPLES_WORKSPACE_NAME), now.toISOString());
  return Number(result.lastInsertRowid);
}

/** Every workspace you can open with what the grid shows on its card. */
export function summarizeWorkspaces(db: DatabaseSync, includeHidden: boolean, videoFamilies: string[]): WorkspaceSummary[] {
  const counts = new Map<number, number>();
  const rows = db
    .prepare('SELECT workspace_id, COUNT(*) AS n FROM generations WHERE trashed_at IS NULL GROUP BY workspace_id')
    .all() as unknown as { workspace_id: number; n: number }[];
  for (const row of rows) counts.set(row.workspace_id, row.n);
  const notVideo = videoFamilies.length === 0 ? '1' : `model_family NOT IN (${videoFamilies.map(() => '?').join(', ')})`;
  const cover = db.prepare(`SELECT image_path FROM generations WHERE workspace_id = ? AND trashed_at IS NULL AND ${notVideo} ORDER BY id DESC LIMIT 1`);
  return listWorkspaces(db, includeHidden).map((w) => {
    const row = cover.get(w.id, ...videoFamilies) as unknown as { image_path: string } | undefined;
    return { ...w, itemCount: counts.get(w.id) ?? 0, coverPath: row?.image_path ?? null };
  });
}

interface ColumnInfo {
  name: string;
}

/**
 * One-time move into workspaces, safe to run on every start:
 *  - everything made before workspaces existed is in the first workspace, "Default" (the `workspace_id`
 *    columns default to it), which inherits the cleanup rules that used to be global;
 *  - items that were individually hidden (the old per-item flag, which no longer exists) are put in a hidden
 *    workspace named "Hidden", so they stay out of sight - except the Styles page's example pictures, which
 *    move to the app's own workspace;
 *  - the flag is then cleared, so this does nothing the next time.
 */
export function migrateToWorkspaces(db: DatabaseSync, legacyCleanup: CleanupSettings, now: Date = new Date()): void {
  const stamp = now.toISOString();
  db.exec('BEGIN');
  try {
    const existing = db.prepare('SELECT COUNT(*) AS n FROM workspaces WHERE system = 0').get() as unknown as { n: number };
    if (existing.n === 0) {
      db.prepare('INSERT INTO workspaces (id, name, theme, hidden, system, cleanup, folder, created_at) VALUES (?, ?, NULL, 0, 0, ?, ?, ?)').run(
        DEFAULT_WORKSPACE_ID,
        DEFAULT_WORKSPACE_NAME,
        JSON.stringify(legacyCleanup),
        nextFolderName(db, DEFAULT_WORKSPACE_NAME),
        stamp
      );
    }

    const columns = db.prepare('PRAGMA table_info(generations)').all() as unknown as ColumnInfo[];
    if (columns.some((c) => c.name === 'hidden')) {
      const hiddenCount = db.prepare('SELECT COUNT(*) AS n FROM generations WHERE hidden = 1').get() as unknown as { n: number };
      if (hiddenCount.n > 0) {
        const examples = db
          .prepare('SELECT COUNT(*) AS n FROM generations WHERE hidden = 1 AND id IN (SELECT generation_id FROM style_samples)')
          .get() as unknown as { n: number };
        if (examples.n > 0) {
          db.prepare('UPDATE generations SET workspace_id = ? WHERE hidden = 1 AND id IN (SELECT generation_id FROM style_samples)').run(
            styleExamplesWorkspaceId(db, now)
          );
        }
        const rest = db.prepare('SELECT COUNT(*) AS n FROM generations WHERE hidden = 1 AND workspace_id != (SELECT COALESCE(MAX(id), -1) FROM workspaces WHERE system = 1)').get() as unknown as { n: number };
        if (rest.n > 0) {
          const named = db.prepare('SELECT id FROM workspaces WHERE system = 0 AND name = ? COLLATE NOCASE').get(HIDDEN_WORKSPACE_NAME) as unknown as { id: number } | undefined;
          const hiddenId =
            named?.id ??
            Number(
              db
                .prepare('INSERT INTO workspaces (name, theme, hidden, system, cleanup, folder, created_at) VALUES (?, NULL, 1, 0, ?, ?, ?)')
                .run(HIDDEN_WORKSPACE_NAME, JSON.stringify(DEFAULT_CLEANUP_SETTINGS), nextFolderName(db, HIDDEN_WORKSPACE_NAME), stamp).lastInsertRowid
            );
          db.prepare('UPDATE generations SET workspace_id = ? WHERE hidden = 1 AND workspace_id != (SELECT COALESCE(MAX(id), -1) FROM workspaces WHERE system = 1)').run(hiddenId);
        }
        db.exec('UPDATE generations SET hidden = 0 WHERE hidden = 1');
      }
    }
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}
