import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GenerationParams, GenerationRecord, SavedPrompt } from '../shared/types';

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
    createdAt,
  };
}

export function listGenerations(db: DatabaseSync): GenerationRecord[] {
  const rows = db.prepare('SELECT * FROM generations ORDER BY id DESC').all() as unknown as GenerationRow[];
  return rows.map(rowToRecord);
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
