import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GenerationKind, GenerationParams, GenerationRecord, GenerationRef, SavedPrompt } from '../shared/types';
import { normalizeName, normalizeTags } from '../shared/promptTags';
import { TIMING_SCHEMA } from './timingStats';

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
  timing_id INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  prompt TEXT NOT NULL,
  negative_prompt TEXT,
  tags TEXT NOT NULL DEFAULT '[]',
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
  if (!columns.some((c) => c.name === 'timing_id')) {
    db.exec('ALTER TABLE generations ADD COLUMN timing_id INTEGER;');
  }
  const promptColumns = db.prepare('PRAGMA table_info(saved_prompts)').all() as unknown as ColumnInfo[];
  if (!promptColumns.some((c) => c.name === 'tags')) {
    db.exec("ALTER TABLE saved_prompts ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';");
  }
}

export function initDatabase(dbPath: string): DatabaseSync {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA journal_mode = WAL;');
  db.exec(SCHEMA);
  db.exec(TIMING_SCHEMA);
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
  // Joined from timing_stats (null when the generation has no recorded timing).
  t_estimate_ms?: number | null;
  t_estimate_generate_ms?: number | null;
  t_actual_ms?: number | null;
  t_generate_ms?: number | null;
  t_load_ms?: number | null;
}

/** A generation plus its timing (joined from the separate timing_stats table). */
const GENERATION_SELECT = `SELECT g.*, t.estimate_ms AS t_estimate_ms, t.estimate_generate_ms AS t_estimate_generate_ms,
  t.actual_ms AS t_actual_ms, t.generate_ms AS t_generate_ms, t.load_ms AS t_load_ms
  FROM generations g LEFT JOIN timing_stats t ON t.id = g.timing_id`;

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
    timing:
      row.t_actual_ms == null
        ? null
        : {
            estimateMs: row.t_estimate_ms ?? null,
            estimateGenerateMs: row.t_estimate_generate_ms ?? null,
            actualMs: row.t_actual_ms,
            generateMs: row.t_generate_ms ?? null,
            loadMs: row.t_load_ms ?? null,
          },
  };
}

export function insertGeneration(
  db: DatabaseSync,
  params: GenerationParams,
  modelFamily: string,
  imagePath: string,
  timingId: number | null = null
): GenerationRecord {
  const createdAt = new Date().toISOString();
  const length = params.length ?? null;
  const stmt = db.prepare(`
    INSERT INTO generations (prompt, negative_prompt, width, height, seed, steps, cfg, length, model_family, image_path, timing_id, created_at)
    VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    timingId,
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
    timing: null,
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
  const cursor = (beforeId === null ? '' : 'AND g.id < ? ') + (favoritesOnly ? 'AND favorite = 1' : '');
  const params: (string | number)[] = [...condition.params];
  if (beforeId !== null) params.push(beforeId);
  params.push(limit);
  const rows = db
    .prepare(`${GENERATION_SELECT} WHERE ${condition.sql} ${cursor} ORDER BY g.id DESC LIMIT ?`)
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

/** Sets the favorite flag, and the file path too when its file was moved along with it. */
export function setGenerationFavorite(db: DatabaseSync, id: number, favorite: boolean, newImagePath?: string): void {
  if (newImagePath === undefined) {
    db.prepare('UPDATE generations SET favorite = ? WHERE id = ?').run(favorite ? 1 : 0, id);
  } else {
    db.prepare('UPDATE generations SET favorite = ?, image_path = ? WHERE id = ?').run(favorite ? 1 : 0, newImagePath, id);
  }
}

export function getGenerationById(db: DatabaseSync, id: number): GenerationRecord | null {
  const row = db.prepare(`${GENERATION_SELECT} WHERE g.id = ?`).get(id) as unknown as GenerationRow | undefined;
  return row ? rowToRecord(row) : null;
}

export function listFavoriteIds(db: DatabaseSync): number[] {
  const rows = db.prepare('SELECT id FROM generations WHERE favorite = 1').all() as unknown as { id: number }[];
  return rows.map((r) => r.id);
}

export function deleteGeneration(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM generations WHERE id = ?').run(id);
}

interface SavedPromptRow {
  id: number;
  name: string | null;
  prompt: string;
  negative_prompt: string | null;
  tags: string;
  created_at: string;
}

function parseTags(json: string | null): string[] {
  try {
    const parsed: unknown = JSON.parse(json ?? '[]');
    return Array.isArray(parsed) ? normalizeTags(parsed.map(String)) : [];
  } catch {
    return [];
  }
}

function rowToSavedPrompt(row: SavedPromptRow): SavedPrompt {
  return {
    id: row.id,
    name: row.name,
    prompt: row.prompt,
    negativePrompt: row.negative_prompt,
    tags: parseTags(row.tags),
    createdAt: row.created_at,
  };
}

export function listSavedPrompts(db: DatabaseSync): SavedPrompt[] {
  const rows = db.prepare('SELECT * FROM saved_prompts ORDER BY id DESC').all() as unknown as SavedPromptRow[];
  return rows.map(rowToSavedPrompt);
}

export function insertSavedPrompt(db: DatabaseSync, name: string, prompt: string, tags: string[]): SavedPrompt {
  const cleanName = normalizeName(name);
  if (!cleanName) throw new Error('A saved prompt needs a name.');
  const cleanTags = normalizeTags(tags);
  const createdAt = new Date().toISOString();
  const result = db
    .prepare('INSERT INTO saved_prompts (name, prompt, negative_prompt, tags, created_at) VALUES (?, ?, NULL, ?, ?)')
    .run(cleanName, prompt, JSON.stringify(cleanTags), createdAt);
  return {
    id: Number(result.lastInsertRowid),
    name: cleanName,
    prompt,
    negativePrompt: null,
    tags: cleanTags,
    createdAt,
  };
}

/** Renames a saved prompt and replaces its tags; the prompt text itself is not edited. */
export function updateSavedPrompt(db: DatabaseSync, id: number, name: string, tags: string[]): SavedPrompt {
  const cleanName = normalizeName(name);
  if (!cleanName) throw new Error('A saved prompt needs a name.');
  db.prepare('UPDATE saved_prompts SET name = ?, tags = ? WHERE id = ?').run(
    cleanName,
    JSON.stringify(normalizeTags(tags)),
    id
  );
  const row = db.prepare('SELECT * FROM saved_prompts WHERE id = ?').get(id) as unknown as SavedPromptRow | undefined;
  if (!row) throw new Error('That prompt no longer exists.');
  return rowToSavedPrompt(row);
}

export function deleteSavedPrompt(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM saved_prompts WHERE id = ?').run(id);
}
