import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GenerationKind, GenerationParams, GenerationRecord, GenerationRef } from '../shared/types';
import { TIMING_SCHEMA } from './timingStats';
import { JOBS_SCHEMA } from './jobStore';
import { IMPORTS_SCHEMA } from './library';
import { ASSEMBLIES_SCHEMA } from './assembly';

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
  hidden INTEGER NOT NULL DEFAULT 0,
  pinned_at TEXT,
  timing_id INTEGER,
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
  if (!columns.some((c) => c.name === 'hidden')) {
    db.exec('ALTER TABLE generations ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0;');
  }
  if (!columns.some((c) => c.name === 'timing_id')) {
    db.exec('ALTER TABLE generations ADD COLUMN timing_id INTEGER;');
  }
  if (!columns.some((c) => c.name === 'pinned_at')) {
    db.exec('ALTER TABLE generations ADD COLUMN pinned_at TEXT;');
  }
}

/**
 * One-time conversion of the old saved-prompts table (a name, prompt text and tags with no picture)
 * into pins on the generations themselves. Each saved prompt pins the newest generation made from
 * exactly that text (preferring one that is not hidden), keeping the saved prompt's date as the pin
 * date so the order survives. A prompt with no matching generation has nothing to pin, so those are
 * written to a text file at `exportPath` first; the table is only dropped once every row is either
 * pinned or exported. With no `exportPath` and something unmatched, or if the file cannot be written,
 * the table is left alone and this is retried on the next start. Returns what was done, or null when
 * nothing was converted.
 */
export function migrateSavedPrompts(db: DatabaseSync, exportPath: string | null): { pinned: number; exported: number } | null {
  const exists = db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'saved_prompts'").get();
  if (!exists) return null;

  const saved = db.prepare('SELECT * FROM saved_prompts ORDER BY id').all() as unknown as {
    name: string | null;
    prompt: string;
    tags: string | null;
    created_at: string;
  }[];
  const findMatch = db.prepare('SELECT id FROM generations WHERE prompt = ? ORDER BY hidden, id DESC LIMIT 1');
  const matched: { id: number; createdAt: string }[] = [];
  const unmatched: typeof saved = [];
  for (const row of saved) {
    const match = findMatch.get(row.prompt) as unknown as { id: number } | undefined;
    if (match) matched.push({ id: match.id, createdAt: row.created_at });
    else unmatched.push(row);
  }

  if (unmatched.length > 0) {
    if (!exportPath) return null;
    const lines = [
      'Saved prompts that had no matching image, kept from before prompts became pinned images.',
      'To keep one, paste it into Generate, make an image, and pin it.',
      '',
    ];
    for (const row of unmatched) {
      let tags: string[] = [];
      try {
        const parsed: unknown = JSON.parse(row.tags ?? '[]');
        if (Array.isArray(parsed)) tags = parsed.map(String);
      } catch {
        // Unreadable tags are not worth blocking the export for.
      }
      lines.push(`# ${row.name?.trim() || '(untitled)'}${tags.length ? `  [${tags.join(', ')}]` : ''}`, row.prompt, '');
    }
    try {
      fs.writeFileSync(exportPath, lines.join('\n'), 'utf8');
    } catch {
      return null;
    }
  }

  // Two saved prompts can land on the same generation: it keeps the older pin date.
  const pin = db.prepare('UPDATE generations SET pinned_at = ? WHERE id = ? AND (pinned_at IS NULL OR pinned_at > ?)');
  db.exec('BEGIN');
  try {
    for (const { id, createdAt } of matched) pin.run(createdAt, id, createdAt);
    db.exec('DROP TABLE saved_prompts');
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
  return { pinned: new Set(matched.map((m) => m.id)).size, exported: unmatched.length };
}

export function initDatabase(dbPath: string): DatabaseSync {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA journal_mode = WAL;');
  db.exec(SCHEMA);
  db.exec(TIMING_SCHEMA);
  db.exec(JOBS_SCHEMA);
  db.exec(IMPORTS_SCHEMA);
  db.exec(ASSEMBLIES_SCHEMA);
  migrateSchema(db);
  // A database in memory has no folder to leave the export in.
  migrateSavedPrompts(db, dbPath === ':memory:' ? null : path.join(path.dirname(dbPath), 'saved-prompts-unpinned.txt'));
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
  hidden: number;
  pinned_at: string | null;
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
    hidden: row.hidden === 1,
    pinned: row.pinned_at !== null,
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
  timingId: number | null = null,
  hidden = false
): GenerationRecord {
  const createdAt = new Date().toISOString();
  const length = params.length ?? null;
  const stmt = db.prepare(`
    INSERT INTO generations (prompt, negative_prompt, width, height, seed, steps, cfg, length, model_family, image_path, hidden, timing_id, created_at)
    VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    hidden ? 1 : 0,
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
    hidden,
    pinned: false,
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

/** Narrows a listing to one file extension (e.g. 'gif'), matched on the stored path. Anything that
 * is not a plain extension is ignored rather than put into a LIKE pattern. */
function extensionCondition(extension: string | null | undefined): { sql: string; params: string[] } {
  if (!extension || !/^[a-z0-9]{1,8}$/i.test(extension)) return { sql: '', params: [] };
  return { sql: 'AND LOWER(image_path) LIKE ? ', params: [`%.${extension.toLowerCase()}`] };
}

/** The Library's extra filters as SQL: favorites only, and hidden ones left out unless asked for. */
function filterSql(favoritesOnly: boolean, showHidden: boolean): string {
  return (favoritesOnly ? 'AND favorite = 1 ' : '') + (showHidden ? '' : 'AND hidden = 0');
}

export function listGenerations(
  db: DatabaseSync,
  videoFamilies: string[],
  kind: GenerationKind,
  limit: number,
  beforeId: number | null,
  favoritesOnly: boolean,
  showHidden: boolean,
  extension: string | null = null
): GenerationRecord[] {
  const condition = kindCondition(videoFamilies, kind);
  const ext = extensionCondition(extension);
  const cursor = ext.sql + (beforeId === null ? '' : 'AND g.id < ? ') + filterSql(favoritesOnly, showHidden);
  const params: (string | number)[] = [...condition.params, ...ext.params];
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
  favoritesOnly: boolean,
  showHidden: boolean,
  extension: string | null = null
): GenerationRef[] {
  const condition = kindCondition(videoFamilies, kind);
  const ext = extensionCondition(extension);
  const rows = db
    .prepare(
      `SELECT id, image_path, favorite, pinned_at IS NOT NULL AS pinned FROM generations WHERE ${condition.sql} ${ext.sql}${filterSql(favoritesOnly, showHidden)} ORDER BY id DESC`
    )
    .all(...condition.params, ...ext.params) as unknown as { id: number; image_path: string; favorite: number; pinned: number }[];
  return rows.map((r) => ({ id: r.id, imagePath: r.image_path, favorite: r.favorite === 1, pinned: r.pinned === 1 }));
}

export function countGenerations(
  db: DatabaseSync,
  videoFamilies: string[],
  favoritesOnly: boolean,
  showHidden: boolean,
  imageExtension: string | null = null
): Record<GenerationKind, number> {
  const count = (kind: GenerationKind): number => {
    const condition = kindCondition(videoFamilies, kind);
    const ext = extensionCondition(kind === 'image' ? imageExtension : null);
    const row = db
      .prepare(`SELECT COUNT(*) AS n FROM generations WHERE ${condition.sql} ${ext.sql}${filterSql(favoritesOnly, showHidden)}`)
      .get(...condition.params, ...ext.params) as unknown as { n: number };
    return row.n;
  };
  return { image: count('image'), video: count('video') };
}

/** The file extensions (lowercase, no dot) present among the images, most common first. */
export function listImageExtensions(db: DatabaseSync, videoFamilies: string[]): string[] {
  const condition = kindCondition(videoFamilies, 'image');
  // The text after the last '.': RTRIM drops the trailing non-dot characters, leaving "...name.",
  // which REPLACE then removes from the path.
  const rows = db
    .prepare(
      `SELECT LOWER(REPLACE(image_path, RTRIM(image_path, REPLACE(image_path, '.', '')), '')) AS ext, COUNT(*) AS n
       FROM generations WHERE ${condition.sql} GROUP BY ext ORDER BY n DESC, ext`
    )
    .all(...condition.params) as unknown as { ext: string }[];
  return rows.map((r) => r.ext).filter((ext) => /^[a-z0-9]{1,8}$/.test(ext));
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

/** Pins a generation as a representative example of its prompt (shown under Library > Prompts), or
 * unpins it. Pinning an already-pinned item keeps its original pin date. */
export function setGenerationPinned(db: DatabaseSync, id: number, pinned: boolean): void {
  if (pinned) {
    db.prepare('UPDATE generations SET pinned_at = ? WHERE id = ? AND pinned_at IS NULL').run(new Date().toISOString(), id);
  } else {
    db.prepare('UPDATE generations SET pinned_at = NULL WHERE id = ?').run(id);
  }
}

/** Every pinned generation, most recently pinned first; hidden ones only when `showHidden`. */
export function listPinnedGenerations(db: DatabaseSync, showHidden: boolean): GenerationRecord[] {
  const rows = db
    .prepare(`${GENERATION_SELECT} WHERE g.pinned_at IS NOT NULL ${showHidden ? '' : 'AND g.hidden = 0'} ORDER BY g.pinned_at DESC, g.id DESC`)
    .all() as unknown as GenerationRow[];
  return rows.map(rowToRecord);
}

export function setGenerationHidden(db: DatabaseSync, id: number, hidden: boolean): void {
  db.prepare('UPDATE generations SET hidden = ? WHERE id = ?').run(hidden ? 1 : 0, id);
}

/** Hides every not-yet-hidden generation whose prompt matches. Only ever hides - never un-hides -
 * so things hidden by hand survive a re-run. Returns how many were checked and newly hidden. */
export function applyHiddenRule(
  db: DatabaseSync,
  matches: (prompt: string) => boolean
): { checked: number; newlyHidden: number } {
  const rows = db.prepare('SELECT id, prompt FROM generations WHERE hidden = 0').all() as unknown as {
    id: number;
    prompt: string;
  }[];
  const hide = db.prepare('UPDATE generations SET hidden = 1 WHERE id = ?');
  let newlyHidden = 0;
  db.exec('BEGIN');
  try {
    for (const row of rows) {
      if (matches(row.prompt)) {
        hide.run(row.id);
        newlyHidden++;
      }
    }
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
  return { checked: rows.length, newlyHidden };
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
