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

const GEN_SELECT = `SELECT g.id, g.prompt, g.width, g.height, g.seed, g.length, g.model_family, g.image_path, g.created_at, j.batch AS batch
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
    .prepare('INSERT INTO imports (origin, kind, path, name, width, height, duration, batch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
    .run(
      item.origin,
      item.kind,
      item.path,
      path.basename(item.path),
      item.width ?? null,
      item.height ?? null,
      item.duration ?? null,
      item.batch ?? null,
      new Date().toISOString()
    );
  return getItem(db, `imp-${Number(result.lastInsertRowid)}`) as ItemView;
}

/** Generated results count as made by an outside client only if the job that produced them came
 * from one. Anything made in the app itself - and anything from before jobs were recorded - does
 * not, so restricting to these keeps a client from seeing work it did not ask for. */
const FROM_CLIENT = "j.source = 'mcp'";

/** `clientOnly`: show only what outside clients created (their generations, imports and assemblies). */
export function getItem(db: DatabaseSync, id: string, options: { clientOnly?: boolean } = {}): ItemView | null {
  const ref = parseItemId(id);
  if (!ref) return null;
  if (ref.table === 'gen') {
    const row = db
      .prepare(`${GEN_SELECT} WHERE g.id = ?${options.clientOnly ? ` AND ${FROM_CLIENT}` : ''}`)
      .get(ref.n) as unknown as GenRow | undefined;
    return row ? genToItem(row) : null;
  }
  const row = db.prepare('SELECT * FROM imports WHERE id = ?').get(ref.n) as unknown as ImpRow | undefined;
  return row ? impToItem(row) : null;
}

export interface ItemFilter {
  kind?: ItemKind;
  origin?: ItemOrigin;
  batch?: string;
  limit?: number;
  /** Only what outside clients created; see getItem. */
  clientOnly?: boolean;
}

/** Newest first across generated and imported items. */
export function listItems(db: DatabaseSync, filter: ItemFilter = {}): ItemView[] {
  const limit = Math.min(Math.max(Math.floor(filter.limit ?? 50) || 0, 1), 200);
  const items: ItemView[] = [];

  if ((filter.origin === undefined || filter.origin === 'generated') && filter.kind !== 'audio') {
    const where: string[] = filter.clientOnly ? [FROM_CLIENT] : [];
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
    const sql = `${GEN_SELECT} ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY g.id DESC LIMIT ${limit}`;
    items.push(...(db.prepare(sql).all(...args) as unknown as GenRow[]).map(genToItem));
  }

  if (filter.origin !== 'generated') {
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
    const sql = `SELECT * FROM imports ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY id DESC LIMIT ${limit}`;
    items.push(...(db.prepare(sql).all(...args) as unknown as ImpRow[]).map(impToItem));
  }

  return items.sort((a, b) => (a.createdAt < b.createdAt ? 1 : a.createdAt > b.createdAt ? -1 : 0)).slice(0, limit);
}
