import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GenerationKind, GenerationParams, GenerationRecord, GenerationRef, SavedPrompt } from '../shared/types';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS generations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT NOT NULL,
  negative_prompt TEXT,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  steps INTEGER NOT NULL,
  cfg REAL NOT NULL,
  length INTEGER,
  model_family TEXT NOT NULL,
  image_path TEXT NOT NULL,
  favorite INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  prompt TEXT NOT NULL,
  negative_prompt TEXT,
  created_at TEXT NOT NULL
);
`;

interface ColumnInfo {
  name: string;
}

/** CREATE TABLE IF NOT EXISTS only covers a brand-new database - an existing one predating
 * the `length` column (added for video families) needs it added in place, or every read/write
 * against it fails with "no such column". Checked rather than blindly run, since re-running
 * ALTER TABLE ADD COLUMN on a column that already exists errors instead of no-op'ing. */
function migrateSchema(db: DatabaseSync): void {
  const columns = db.prepare('PRAGMA table_info(generations)').all() as unknown as ColumnInfo[];
  if (!columns.some((c) => c.name === 'length')) {
    db.exec('ALTER TABLE generations ADD COLUMN length INTEGER;');
  }
  if (!columns.some((c) => c.name === 'favorite')) {
    db.exec('ALTER TABLE generations ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0;');
  }
}

export function initDatabase(dbPath: string): DatabaseSync {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA journal_mode = WAL;');
  db.exec(SCHEMA);
  migrateSchema(db);
  return db;
}

interface GenerationRow {
  id: number;
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  length: number | null;
  model_family: string;
  image_path: string;
  favorite: number;
  created_at: string;
}

function rowToRecord(row: GenerationRow): GenerationRecord {
  return {
    id: row.id,
    prompt: row.prompt,
    negativePrompt: row.negative_prompt,
    width: row.width,
    height: row.height,
    seed: row.seed,
    steps: row.steps,
    cfg: row.cfg,
    length: row.length,
    modelFamily: row.model_family,
    imagePath: row.image_path,
    favorite: row.favorite === 1,
    createdAt: row.created_at,
  };
}

export function insertGeneration(
  db: DatabaseSync,
  params: GenerationParams,
  modelFamily: string,
  imagePath: string
): GenerationRecord {
  const createdAt = new Date().toISOString();
  const length = params.length ?? null;
  const stmt = db.prepare(`
    INSERT INTO generations (prompt, negative_prompt, width, height, seed, steps, cfg, length, model_family, image_path, created_at)
    VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const result = stmt.run(
    params.prompt,
    params.width,
    params.height,
    params.seed,
    params.steps,
    params.cfg,
    length,
    modelFamily,
    imagePath,
    createdAt
  );
  return {
    id: Number(result.lastInsertRowid),
    prompt: params.prompt,
    negativePrompt: null,
    width: params.width,
    height: params.height,
    seed: params.seed,
    steps: params.steps,
    cfg: params.cfg,
    length,
    modelFamily,
    imagePath,
    favorite: false,
    createdAt,
  };
}

/** SQL condition selecting the image or video half of the table. Families aren't a column of
 * their own kind (FAMILY_KIND lives in shared code), so callers pass the video family list;
 * anything not in it - including a family this build doesn't know - counts as an image. */
function kindCondition(videoFamilies: string[], kind: GenerationKind): { sql: string; params: string[] } {
  if (videoFamilies.length === 0) return { sql: kind === 'video' ? '0' : '1', params: [] };
  const marks = videoFamilies.map(() => '?').join(', ');
  return { sql: `model_family ${kind === 'video' ? 'IN' : 'NOT IN'} (${marks})`, params: videoFamilies };
}

export function listGenerations(
  db: DatabaseSync,
  videoFamilies: string[],
  kind: GenerationKind,
  limit: number,
  beforeId: number | null,
  favoritesOnly: boolean
): GenerationRecord[] {
  const condition = kindCondition(videoFamilies, kind);
  const cursor = (beforeId === null ? '' : 'AND id < ? ') + (favoritesOnly ? 'AND favorite = 1' : '');
  const params: (string | number)[] = [...condition.params];
  if (beforeId !== null) params.push(beforeId);
  params.push(limit);
  const rows = db
    .prepare(`SELECT * FROM generations WHERE ${condition.sql} ${cursor} ORDER BY id DESC LIMIT ?`)
    .all(...params) as unknown as GenerationRow[];
  return rows.map(rowToRecord);
}

export function listGenerationRefs(
  db: DatabaseSync,
  videoFamilies: string[],
  kind: GenerationKind,
  favoritesOnly: boolean
): GenerationRef[] {
  const condition = kindCondition(videoFamilies, kind);
  const rows = db
    .prepare(
      `SELECT id, image_path, favorite FROM generations WHERE ${condition.sql} ${favoritesOnly ? 'AND favorite = 1' : ''} ORDER BY id DESC`
    )
    .all(...condition.params) as unknown as { id: number; image_path: string; favorite: number }[];
  return rows.map((r) => ({ id: r.id, imagePath: r.image_path, favorite: r.favorite === 1 }));
}

export function countGenerations(
  db: DatabaseSync,
  videoFamilies: string[],
  favoritesOnly: boolean
): Record<GenerationKind, number> {
  const count = (kind: GenerationKind): number => {
    const condition = kindCondition(videoFamilies, kind);
    const row = db
      .prepare(`SELECT COUNT(*) AS n FROM generations WHERE ${condition.sql} ${favoritesOnly ? 'AND favorite = 1' : ''}`)
      .get(...condition.params) as unknown as { n: number };
    return row.n;
  };
  return { image: count('image'), video: count('video') };
}

/**
 * One-time tidy-up for output saved before images and videos got their own folders: any file
 * that still sits directly in `legacyDir` is moved into `imagesDir` or `videosDir` (by its row's
 * model family) and its row's path updated. Each file is moved (a same-volume rename) before its
 * row is touched, and moved back if the row update fails, so a row never points at a file that
 * isn't there. Files that are missing, already elsewhere (e.g. under a previously relocated
 * database), or would collide with an existing file are left alone. Returns how many were moved.
 */
export function moveLegacyOutput(
  db: DatabaseSync,
  videoFamilies: string[],
  legacyDir: string,
  imagesDir: string,
  videosDir: string
): number {
  const rows = db.prepare('SELECT id, model_family, image_path FROM generations').all() as unknown as {
    id: number;
    model_family: string;
    image_path: string;
  }[];

  const sourceDir = path.resolve(legacyDir);
  let moved = 0;
  for (const row of rows) {
    const from = path.resolve(row.image_path);
    if (path.dirname(from) !== sourceDir || !fs.existsSync(from)) continue;
    const targetDir = videoFamilies.includes(row.model_family) ? videosDir : imagesDir;
    const to = path.join(targetDir, path.basename(from));
    if (fs.existsSync(to)) continue;
    try {
      fs.mkdirSync(targetDir, { recursive: true });
      fs.renameSync(from, to);
    } catch {
      continue;
    }
    try {
      db.prepare('UPDATE generations SET image_path = ? WHERE id = ?').run(to, row.id);
      moved++;
    } catch {
      fs.renameSync(to, from);
    }
  }
  return moved;
}

export function setGenerationFavorite(db: DatabaseSync, id: number, favorite: boolean): void {
  db.prepare('UPDATE generations SET favorite = ? WHERE id = ?').run(favorite ? 1 : 0, id);
}

export function deleteGeneration(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM generations WHERE id = ?').run(id);
}

interface SavedPromptRow {
  id: number;
  name: string | null;
  prompt: string;
  negative_prompt: string | null;
  created_at: string;
}

function rowToSavedPrompt(row: SavedPromptRow): SavedPrompt {
  return {
    id: row.id,
    name: row.name,
    prompt: row.prompt,
    negativePrompt: row.negative_prompt,
    createdAt: row.created_at,
  };
}

export function listSavedPrompts(db: DatabaseSync): SavedPrompt[] {
  const rows = db.prepare('SELECT * FROM saved_prompts ORDER BY id DESC').all() as unknown as SavedPromptRow[];
  return rows.map(rowToSavedPrompt);
}

export function insertSavedPrompt(db: DatabaseSync, name: string | null, prompt: string): SavedPrompt {
  const createdAt = new Date().toISOString();
  const stmt = db.prepare(`
    INSERT INTO saved_prompts (name, prompt, negative_prompt, created_at)
    VALUES (?, ?, NULL, ?)
  `);
  const result = stmt.run(name, prompt, createdAt);
  return { id: Number(result.lastInsertRowid), name, prompt, negativePrompt: null, createdAt };
}

export function deleteSavedPrompt(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM saved_prompts WHERE id = ?').run(id);
}
