import { DatabaseSync } from 'node:sqlite';
import * as path from 'path';
import { FAMILY_KIND } from '../shared/types';

/**
 * Files that did not come out of a generation but that outside clients need to refer to by id:
 * images imported from a folder (sources for video), audio for a soundtrack, and the videos the
 * app assembles itself. Files are referenced in place. Generated results stay in `generations`;
 * both are presented uniformly as items (ids "gen-<n>" and "imp-<n>").
 */
export const IMPORTS_SCHEMA = `
CREATE TABLE IF NOT EXISTS imports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  origin TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  duration REAL,
  batch TEXT,
  workspace_id INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
`;

export type ItemKind = 'image' | 'video' | 'audio';
export type ItemOrigin = 'generated' | 'imported' | 'assembled';

export interface ItemView {
  id: string;
  kind: ItemKind;
  origin: ItemOrigin;
  name: string;
  path: string;
  width: number | null;
  height: number | null;
  /** Length in seconds, when known (generated video; assembled/imported media). */
  seconds: number | null;
  batch: string | null;
  createdAt: string;
  /** Generated items only. */
  prompt?: string;
  family?: string;
  seed?: number;
  /** Generated items only: kept as a favorite (its file lives in the favorites folder) / pinned as the example of its prompt. */
  favorite?: boolean;
  pinned?: boolean;
}

export const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'];
export const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.webm', '.mkv', '.m4v'];
export const AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus'];

export function kindForExtension(file: string): ItemKind | null {
  const ext = path.extname(file).toLowerCase();
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
  if (VIDEO_EXTENSIONS.includes(ext)) return 'video';
  if (AUDIO_EXTENSIONS.includes(ext)) return 'audio';
  return null;
}

const VIDEO_FPS = 16;

export function parseItemId(id: string): { table: 'gen' | 'imp'; n: number } | null {
  const m = /^(gen|imp)-(\d+)$/.exec(id.trim());
  return m ? { table: m[1] as 'gen' | 'imp', n: Number(m[2]) } : null;
}

interface GenRow {
  id: number;
  prompt: string;
  width: number;
  height: number;
  seed: number;
  length: number | null;
  model_family: string;
  image_path: string;
  created_at: string;
  batch: string | null;
  favorite: number;
  pinned_at: string | null;
}

interface ImpRow {
  id: number;
  origin: string;
  kind: string;
  path: string;
  name: string;
  width: number | null;
  height: number | null;
  duration: number | null;
  batch: string | null;
  created_at: string;
}

const GEN_SELECT = `SELECT g.id, g.prompt, g.width, g.height, g.seed, g.length, g.model_family, g.image_path, g.created_at, g.favorite, g.pinned_at, j.batch AS batch
  FROM generations g LEFT JOIN jobs j ON j.generation_id = g.id`;

function genToItem(row: GenRow): ItemView {
  const kind: ItemKind = FAMILY_KIND[row.model_family] === 'video' ? 'video' : 'image';
  return {
    id: `gen-${row.id}`,
    kind,
    origin: 'generated',
    name: path.basename(row.image_path),
    path: row.image_path,
    width: row.width,
    height: row.height,
    seconds: kind === 'video' && row.length ? Math.round(((row.length - 1) / VIDEO_FPS) * 100) / 100 : null,
    batch: row.batch,
    createdAt: row.created_at,
    prompt: row.prompt,
    family: row.model_family,
    seed: row.seed,
    favorite: row.favorite === 1,
    pinned: row.pinned_at !== null,
  };
}

function impToItem(row: ImpRow): ItemView {
  return {
    id: `imp-${row.id}`,
    kind: row.kind as ItemKind,
    origin: row.origin as ItemOrigin,
    name: row.name,
    path: row.path,
    width: row.width,
    height: row.height,
    seconds: row.duration,
    batch: row.batch,
    createdAt: row.created_at,
  };
}

export interface NewImport {
  path: string;
  kind: ItemKind;
  origin: 'imported' | 'assembled';
  width?: number | null;
  height?: number | null;
  duration?: number | null;
  batch?: string | null;
  /** The workspace the file is added to. */
  workspaceId?: number;
}

/** Adds a file, or returns the existing entry when this path was imported before (a given batch
 * label then updates it, so re-importing a folder into a new batch regroups it). */
export function upsertImport(db: DatabaseSync, item: NewImport): ItemView {
  const existing = db.prepare('SELECT * FROM imports WHERE path = ?').get(item.path) as unknown as ImpRow | undefined;
  if (existing) {
    if (item.batch) db.prepare('UPDATE imports SET batch = ? WHERE id = ?').run(item.batch, existing.id);
    return getItem(db, `imp-${existing.id}`) as ItemView;
  }
  const result = db
    .prepare('INSERT INTO imports (origin, kind, path, name, width, height, duration, batch, workspace_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)')
    .run(
      item.origin,
      item.kind,
      item.path,
      path.basename(item.path),
      item.width ?? null,
      item.height ?? null,
      item.duration ?? null,
      item.batch ?? null,
      item.workspaceId ?? 1,
      new Date().toISOString()
    );
  return getItem(db, `imp-${Number(result.lastInsertRowid)}`) as ItemView;
}

/** Generated results count as made by an outside client only if the job that produced them came
 * from one. Anything made in the app itself - and anything from before jobs were recorded - does
 * not, so restricting to these keeps a client from seeing work it did not ask for. */
const FROM_CLIENT = "j.source = 'mcp'";

/** `clientOnly`: show only what outside clients created (their generations, imports and assemblies). */
export function getItem(db: DatabaseSync, id: string, options: { clientOnly?: boolean; excludeWorkspaceIds?: number[] } = {}): ItemView | null {
  const ref = parseItemId(id);
  if (!ref) return null;
  const excluded = options.excludeWorkspaceIds ?? [];
  if (ref.table === 'gen') {
    // Anything in the Trash no longer exists as far as clients are concerned.
    const row = db
      .prepare(`${GEN_SELECT} WHERE g.id = ? AND g.trashed_at IS NULL${options.clientOnly ? ` AND ${FROM_CLIENT}` : ''}${notInWorkspaces('g.workspace_id', excluded)}`)
      .get(ref.n) as unknown as GenRow | undefined;
    return row ? genToItem(row) : null;
  }
  const row = db.prepare(`SELECT * FROM imports WHERE id = ?${notInWorkspaces('workspace_id', excluded)}`).get(ref.n) as unknown as ImpRow | undefined;
  return row ? impToItem(row) : null;
}

/** ` AND <column> NOT IN (...)` for the given workspace ids (they are numbers made by the app, never caller text), or nothing when there are none. */
function notInWorkspaces(column: string, ids: number[]): string {
  const safe = ids.filter((n) => Number.isInteger(n));
  return safe.length ? ` AND ${column} NOT IN (${safe.join(',')})` : '';
}

/** Whether a path already belongs to an import in one of these workspaces (used to keep a client from touching an import it cannot see). */
export function importPathInWorkspaces(db: DatabaseSync, file: string, ids: number[]): boolean {
  if (ids.length === 0) return false;
  const safe = ids.filter((n) => Number.isInteger(n));
  if (safe.length === 0) return false;
  return db.prepare(`SELECT 1 FROM imports WHERE path = ? AND workspace_id IN (${safe.join(',')})`).get(file) !== undefined;
}

export interface ItemFilter {
  kind?: ItemKind;
  origin?: ItemOrigin;
  batch?: string;
  limit?: number;
  /** Only what outside clients created; see getItem. */
  clientOnly?: boolean;
  /** Only items in this workspace. */
  workspaceId?: number;
  /** Leave out items of these workspaces (the hidden ones, for an outside client). */
  excludeWorkspaceIds?: number[];
  /** Only favorites (true) or only non-favorites (false). Imports and assemblies have no flags, so any flag filter leaves them out. */
  favorite?: boolean;
  /** Only pinned items (true) or only unpinned ones (false); see `favorite`. */
  pinned?: boolean;
}

/** Newest first across generated and imported items. */
export function listItems(db: DatabaseSync, filter: ItemFilter = {}): ItemView[] {
  return collectItems(db, filter, Math.min(Math.max(Math.floor(filter.limit ?? 50) || 0, 1), 200));
}

function collectItems(db: DatabaseSync, filter: ItemFilter, limit: number): ItemView[] {
  const items: ItemView[] = [];

  if ((filter.origin === undefined || filter.origin === 'generated') && filter.kind !== 'audio') {
    const where: string[] = ['g.trashed_at IS NULL'];
    if (filter.clientOnly) where.push(FROM_CLIENT);
    const args: (string | number)[] = [];
    const videoFamilies = Object.keys(FAMILY_KIND).filter((f) => FAMILY_KIND[f] === 'video');
    if (filter.kind === 'video') {
      where.push(`g.model_family IN (${videoFamilies.map(() => '?').join(',')})`);
      args.push(...videoFamilies);
    } else if (filter.kind === 'image') {
      where.push(`g.model_family NOT IN (${videoFamilies.map(() => '?').join(',')})`);
      args.push(...videoFamilies);
    }
    if (filter.batch !== undefined) {
      where.push('j.batch = ?');
      args.push(filter.batch);
    }
    if (filter.workspaceId !== undefined) {
      where.push('g.workspace_id = ?');
      args.push(filter.workspaceId);
    }
    const genExcluded = notInWorkspaces('g.workspace_id', filter.excludeWorkspaceIds ?? []);
    if (genExcluded) where.push(genExcluded.slice(' AND '.length));
    if (filter.favorite !== undefined) where.push(filter.favorite ? 'g.favorite = 1' : 'g.favorite = 0');
    if (filter.pinned !== undefined) where.push(filter.pinned ? 'g.pinned_at IS NOT NULL' : 'g.pinned_at IS NULL');
    const sql = `${GEN_SELECT} ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY g.id DESC LIMIT ${limit}`;
    items.push(...(db.prepare(sql).all(...args) as unknown as GenRow[]).map(genToItem));
  }

  if (filter.origin !== 'generated' && filter.favorite === undefined && filter.pinned === undefined) {
    const where: string[] = [];
    const args: (string | number)[] = [];
    if (filter.kind) {
      where.push('kind = ?');
      args.push(filter.kind);
    }
    if (filter.origin) {
      where.push('origin = ?');
      args.push(filter.origin);
    }
    if (filter.batch !== undefined) {
      where.push('batch = ?');
      args.push(filter.batch);
    }
    if (filter.workspaceId !== undefined) {
      where.push('workspace_id = ?');
      args.push(filter.workspaceId);
    }
    const impExcluded = notInWorkspaces('workspace_id', filter.excludeWorkspaceIds ?? []);
    if (impExcluded) where.push(impExcluded.slice(' AND '.length));
    const sql = `SELECT * FROM imports ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY id DESC LIMIT ${limit}`;
    items.push(...(db.prepare(sql).all(...args) as unknown as ImpRow[]).map(impToItem));
  }

  return items.sort((a, b) => (a.createdAt < b.createdAt ? 1 : a.createdAt > b.createdAt ? -1 : 0)).slice(0, limit);
}

/** The most items grouping will look through; a group list built from more says `truncated`. */
export const GROUPING_ITEM_CAP = 5000;

export type GroupBy = 'prompt' | 'batch';

export interface ItemGroup {
  /** The prompt (group_by prompt) or the batch label (null: no batch). */
  key: string | null;
  count: number;
  favorites: number;
  pinned: number;
  /** The pinned item if there is one, else a favorite, else the newest - like a stack's cover in the app's Library. */
  cover: ItemView;
  /** Every item of the group, newest first (at most `GROUP_ITEM_ID_LIMIT`). */
  itemIds: string[];
  newestAt: string;
}

export const GROUP_ITEM_ID_LIMIT = 200;

/**
 * The items passing `filter`, collected into groups by identical prompt (generated items only - only they have one) or by batch
 * label, newest group first. The filter applies to items first, so a group only counts what passes it. Looks through at most
 * GROUPING_ITEM_CAP items (the newest); `truncated` says it hit that.
 */
export function groupItems(db: DatabaseSync, filter: ItemFilter, by: GroupBy): { groups: ItemGroup[]; truncated: boolean } {
  const base: ItemFilter = by === 'prompt' ? { ...filter, origin: 'generated', limit: undefined } : { ...filter, limit: undefined };
  const items = collectItems(db, base, GROUPING_ITEM_CAP);
  const groups = new Map<string, ItemView[]>();
  for (const item of items) {
    const key = by === 'prompt' ? (item.prompt ?? '') : (item.batch ?? '\u0000');
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  const rank = (i: ItemView) => (i.pinned ? 2 : i.favorite ? 1 : 0);
  const out: ItemGroup[] = [...groups.entries()].map(([key, members]) => {
    // `members` is newest first, so on a tie the earliest element (the newest) wins.
    const cover = members.reduce((best, m) => (rank(m) > rank(best) ? m : best), members[0]);
    return {
      key: by === 'batch' && key === '\u0000' ? null : key,
      count: members.length,
      favorites: members.filter((m) => m.favorite).length,
      pinned: members.filter((m) => m.pinned).length,
      cover,
      itemIds: members.slice(0, GROUP_ITEM_ID_LIMIT).map((m) => m.id),
      newestAt: members[0].createdAt,
    };
  });
  out.sort((a, b) => (a.newestAt < b.newestAt ? 1 : a.newestAt > b.newestAt ? -1 : 0));
  const limit = Math.min(Math.max(Math.floor(filter.limit ?? 50) || 0, 1), 200);
  return { groups: out.slice(0, limit), truncated: items.length >= GROUPING_ITEM_CAP };
}
