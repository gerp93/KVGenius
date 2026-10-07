import { DatabaseSync } from 'node:sqlite';
import { ModelProfile, ModelProfileInput, validateProfileInput } from '../shared/modelProfiles';

/**
 * Saved model variants (see shared/modelProfiles.ts). Names are unique ignoring case so a client can ask
 * for one by name (the MCP `model` argument). The built-in model is not a row - it is the shipped template.
 */
export const MODEL_PROFILES_SCHEMA = `
CREATE TABLE IF NOT EXISTS model_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  family TEXT NOT NULL,
  name TEXT NOT NULL COLLATE NOCASE,
  files TEXT NOT NULL,
  steps INTEGER NOT NULL,
  cfg REAL NOT NULL,
  sampler TEXT NOT NULL,
  scheduler TEXT NOT NULL,
  shift REAL NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_profiles_name ON model_profiles (name COLLATE NOCASE);
`;

interface ProfileRow {
  id: number;
  family: string;
  name: string;
  files: string;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  shift: number;
  created_at: string;
  updated_at: string;
}

function rowToProfile(row: ProfileRow): ModelProfile {
  let files: Record<string, string> = {};
  try {
    files = JSON.parse(row.files) as Record<string, string>;
  } catch {
    // A damaged row shows with no files; the editor makes the user choose them again.
  }
  return {
    id: row.id,
    family: row.family,
    name: row.name,
    files,
    sampler: { steps: row.steps, cfg: row.cfg, sampler: row.sampler, scheduler: row.scheduler, shift: row.shift },
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function isNameTaken(err: unknown): boolean {
  return err instanceof Error && /UNIQUE constraint failed/i.test(err.message);
}

/** Alphabetical (ignoring case), which is also the order of the Generate page's dropdown. */
export function listModelProfiles(db: DatabaseSync): ModelProfile[] {
  return (db.prepare('SELECT * FROM model_profiles ORDER BY name COLLATE NOCASE, id').all() as unknown as ProfileRow[]).map(rowToProfile);
}

export function getModelProfile(db: DatabaseSync, id: number): ModelProfile | null {
  const row = db.prepare('SELECT * FROM model_profiles WHERE id = ?').get(id) as unknown as ProfileRow | undefined;
  return row ? rowToProfile(row) : null;
}

export function findModelProfileByName(db: DatabaseSync, name: string): ModelProfile | null {
  const row = db.prepare('SELECT * FROM model_profiles WHERE name = ? COLLATE NOCASE').get(name.trim()) as unknown as ProfileRow | undefined;
  return row ? rowToProfile(row) : null;
}

/** Saves a new profile, or updates the one with `id`. Throws a message fit to show the user. */
export function saveModelProfile(db: DatabaseSync, input: ModelProfileInput, id: number | null = null): ModelProfile {
  const checked = validateProfileInput(input);
  if (!checked.ok) throw new Error(checked.message);
  const { family, name, files, sampler } = checked.value;
  const now = new Date().toISOString();
  try {
    if (id === null) {
      const result = db
        .prepare(
          `INSERT INTO model_profiles (family, name, files, steps, cfg, sampler, scheduler, shift, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(family, name, JSON.stringify(files), sampler.steps, sampler.cfg, sampler.sampler, sampler.scheduler, sampler.shift, now, now);
      return getModelProfile(db, Number(result.lastInsertRowid)) as ModelProfile;
    }
    const result = db
      .prepare('UPDATE model_profiles SET family = ?, name = ?, files = ?, steps = ?, cfg = ?, sampler = ?, scheduler = ?, shift = ?, updated_at = ? WHERE id = ?')
      .run(family, name, JSON.stringify(files), sampler.steps, sampler.cfg, sampler.sampler, sampler.scheduler, sampler.shift, now, id);
    if (Number(result.changes) === 0) throw new Error('That model no longer exists.');
    return getModelProfile(db, id) as ModelProfile;
  } catch (err) {
    if (isNameTaken(err)) throw new Error(`A model named "${name}" already exists.`);
    throw err;
  }
}

/** Deleting a profile never touches past generations: they keep the files and settings they were made with. */
export function deleteModelProfile(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM model_profiles WHERE id = ?').run(id);
}
