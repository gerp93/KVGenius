import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { GenerationKind, GenerationParams, GenerationRecord, GenerationRef, LibraryListOptions } from '../shared/types';
import { TIMING_SCHEMA } from './timingStats';
import { JOBS_SCHEMA } from './jobStore';
import { IMPORTS_SCHEMA } from './library';
import { ASSEMBLIES_SCHEMA } from './assembly';
import { STYLES_SCHEMA } from './styles';

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
  source_image_path TEXT,
  trashed_at TEXT,
  trash_from TEXT,
  style_name TEXT,
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
  if (!columns.some((c) => c.name === 'source_image_path')) {
    db.exec('ALTER TABLE generations ADD COLUMN source_image_path TEXT;');
  }
  // Trash: when an item was moved there, and where its file came from so it can be put back.
  if (!columns.some((c) => c.name === 'trashed_at')) {
    db.exec('ALTER TABLE generations ADD COLUMN trashed_at TEXT;');
  }
  if (!columns.some((c) => c.name === 'trash_from')) {
    db.exec('ALTER TABLE generations ADD COLUMN trash_from TEXT;');
  }
  // The name of the style (see styles.ts) a prompt was combined with - display only, the stored prompt is already combined.
  if (!columns.some((c) => c.name === 'style_name')) {
    db.exec('ALTER TABLE generations ADD COLUMN style_name TEXT;');
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
  db.exec(STYLES_SCHEMA);
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
  source_image_path: string | null;
  trashed_at: string | null;
  style_name: string | null;
  created_at: string;
  // Only in a listing grouped by prompt.
  group_count?: number;
  group_newest?: number;
  // Joined from timing_stats (null when the generation has no recorded timing).
  t_estimate_ms?: number | null;
  t_estimate_generate_ms?: number | null;
  t_actual_ms?: number | null;
  t_generate_ms?: number | null;
  t_load_ms?: number | null;
}

const TIMING_COLUMNS = `t.estimate_ms AS t_estimate_ms, t.estimate_generate_ms AS t_estimate_generate_ms,
  t.actual_ms AS t_actual_ms, t.generate_ms AS t_generate_ms, t.load_ms AS t_load_ms`;

/** A generation plus its timing (joined from the separate timing_stats table). */
const GENERATION_SELECT = `SELECT g.*, ${TIMING_COLUMNS}
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
    sourceImagePath: row.source_image_path,
    trashedAt: row.trashed_at,
    styleName: row.style_name ?? null,
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
    ...(row.group_count !== undefined ? { groupCount: row.group_count, groupNewestId: row.group_newest } : {}),
  };
}

export function insertGeneration(
  db: DatabaseSync,
  params: GenerationParams,
  modelFamily: string,
  imagePath: string,
  timingId: number | null = null,
  hidden = false,
  /** A video's kept copy of its source image (see sourceImages.ts), so it can be re-run in place. */
  sourceImagePath: string | null = null
): GenerationRecord {
  const createdAt = new Date().toISOString();
  const length = params.length ?? null;
  const styleName = params.styleName?.trim() || null;
  const stmt = db.prepare(`
    INSERT INTO generations (prompt, negative_prompt, width, height, seed, steps, cfg, length, model_family, image_path, hidden, source_image_path, style_name, timing_id, created_at)
    VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    sourceImagePath,
    styleName,
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
    sourceImagePath,
    trashedAt: null,
    styleName,
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

/** The Library's filters as SQL: anything in the Trash is always left out, favorites only on request,
 * and hidden ones left out unless asked for. */
function filterSql(favoritesOnly: boolean, showHidden: boolean): string {
  return 'AND trashed_at IS NULL ' + (favoritesOnly ? 'AND favorite = 1 ' : '') + (showHidden ? '' : 'AND hidden = 0');
}

/** Narrows a listing to the items whose prompt is exactly `prompt` (what opening a stack shows). */
function promptCondition(prompt: string | null | undefined): { sql: string; params: string[] } {
  return typeof prompt === 'string' ? { sql: 'AND prompt = ? ', params: [prompt] } : { sql: '', params: [] };
}

/** How many pictures a stack's card cycles through (a stack of hundreds would otherwise load them all). */
const STACK_PREVIEW_LIMIT = 12;

/**
 * One cover per distinct prompt, newest stack first. Items whose prompt is exactly the same (not
 * similar - exactly) collapse into a stack; the cover is the pinned item if there is one, else a
 * favorite, else the newest, and carries how many items the stack holds. Stacks are ordered and paged
 * by their newest item (`beforeNewest` is the previous page's last `groupNewestId`), so a stack with an
 * old pinned cover still sits where its latest generation puts it. The Library's filters apply to the
 * items first, so a stack only counts the items that pass them.
 */
function listPromptStacks(
  db: DatabaseSync,
  where: string,
  whereParams: (string | number)[],
  limit: number,
  beforeNewest: number | null
): GenerationRecord[] {
  const params = [...whereParams];
  if (beforeNewest !== null) params.push(beforeNewest);
  params.push(limit);
  const rows = db
    .prepare(
      `SELECT g.*, ${TIMING_COLUMNS}
       FROM (
         SELECT *,
           COUNT(*) OVER (PARTITION BY prompt) AS group_count,
           MAX(id) OVER (PARTITION BY prompt) AS group_newest,
           ROW_NUMBER() OVER (PARTITION BY prompt ORDER BY (pinned_at IS NOT NULL) DESC, favorite DESC, id DESC) AS rn
         FROM generations WHERE ${where}
       ) g LEFT JOIN timing_stats t ON t.id = g.timing_id
       WHERE g.rn = 1 ${beforeNewest === null ? '' : 'AND g.group_newest < ?'}
       ORDER BY g.group_newest DESC LIMIT ?`
    )
    .all(...params) as unknown as GenerationRow[];
  const records = rows.map(rowToRecord);

  // What each stack's card cycles through: the cover first, then the stack's newest other items.
  const stacks = records.filter((r) => (r.groupCount ?? 1) > 1);
  if (stacks.length > 0) {
    const marks = stacks.map(() => '?').join(', ');
    const members = db
      .prepare(
        `SELECT prompt, image_path FROM (
           SELECT prompt, image_path,
             ROW_NUMBER() OVER (PARTITION BY prompt ORDER BY id DESC) AS rn
           FROM generations WHERE ${where} AND prompt IN (${marks})
         ) WHERE rn <= ? ORDER BY prompt, rn`
      )
      .all(...whereParams, ...stacks.map((r) => r.prompt), STACK_PREVIEW_LIMIT + 1) as unknown as { prompt: string; image_path: string }[];
    const byPrompt = new Map<string, string[]>();
    for (const m of members) byPrompt.set(m.prompt, [...(byPrompt.get(m.prompt) ?? []), m.image_path]);
    for (const stack of stacks) {
      const others = (byPrompt.get(stack.prompt) ?? []).filter((p) => p !== stack.imagePath);
      stack.groupPreviewPaths = [stack.imagePath, ...others].slice(0, STACK_PREVIEW_LIMIT);
    }
  }
  return records;
}

export function listGenerations(
  db: DatabaseSync,
  videoFamilies: string[],
  kind: GenerationKind,
  limit: number,
  beforeId: number | null,
  favoritesOnly: boolean,
  showHidden: boolean,
  extension: string | null = null,
  options: LibraryListOptions = {}
): GenerationRecord[] {
  const condition = kindCondition(videoFamilies, kind);
  const ext = extensionCondition(extension);
  const prompt = promptCondition(options.prompt);

  if (options.grouped && prompt.sql === '') {
    return listPromptStacks(
      db,
      `${condition.sql} ${ext.sql}${filterSql(favoritesOnly, showHidden)}`,
      [...condition.params, ...ext.params],
      limit,
      beforeId
    );
  }

  const cursor = ext.sql + prompt.sql + (beforeId === null ? '' : 'AND g.id < ? ') + filterSql(favoritesOnly, showHidden);
  const params: (string | number)[] = [...condition.params, ...ext.params, ...prompt.params];
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
  extension: string | null = null,
  options: LibraryListOptions = {}
): GenerationRef[] {
  const condition = kindCondition(videoFamilies, kind);
  const ext = extensionCondition(extension);
  const prompt = promptCondition(options.prompt);
  const rows = db
    .prepare(
      `SELECT id, image_path, favorite, pinned_at IS NOT NULL AS pinned FROM generations WHERE ${condition.sql} ${ext.sql}${prompt.sql}${filterSql(favoritesOnly, showHidden)} ORDER BY id DESC`
    )
    .all(...condition.params, ...ext.params, ...prompt.params) as unknown as { id: number; image_path: string; favorite: number; pinned: number }[];
  return rows.map((r) => ({ id: r.id, imagePath: r.image_path, favorite: r.favorite === 1, pinned: r.pinned === 1 }));
}

export function countGenerations(
  db: DatabaseSync,
  videoFamilies: string[],
  favoritesOnly: boolean,
  showHidden: boolean,
  imageExtension: string | null = null,
  options: LibraryListOptions = {}
): Record<GenerationKind, number> {
  const prompt = promptCondition(options.prompt);
  // Grouped, the number is stacks (distinct prompts); opened onto one prompt, it is that prompt's items.
  const grouped = options.grouped === true && prompt.sql === '';
  const count = (kind: GenerationKind): number => {
    const condition = kindCondition(videoFamilies, kind);
    const ext = extensionCondition(kind === 'image' ? imageExtension : null);
    const row = db
      .prepare(
        `SELECT ${grouped ? 'COUNT(DISTINCT prompt)' : 'COUNT(*)'} AS n FROM generations WHERE ${condition.sql} ${ext.sql}${prompt.sql}${filterSql(favoritesOnly, showHidden)}`
      )
      .get(...condition.params, ...ext.params, ...prompt.params) as unknown as { n: number };
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
       FROM generations WHERE ${condition.sql} AND trashed_at IS NULL GROUP BY ext ORDER BY n DESC, ext`
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

/** How many pinned generations (in the Library) have exactly this prompt - the size of the group
 * they form under Library > Prompts, where one tile cycles through them. */
export function countPinnedWithPrompt(db: DatabaseSync, prompt: string): number {
  const row = db
    .prepare('SELECT COUNT(*) AS n FROM generations WHERE pinned_at IS NOT NULL AND trashed_at IS NULL AND prompt = ?')
    .get(prompt) as unknown as { n: number };
  return row.n;
}

/** The settings that make a run produce the same output. */
export interface DuplicateQuery {
  prompt: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  /** Video only: frame count. */
  length?: number | null;
  /** Video only: the kept copy of the source image. Videos from a different source are different output. */
  sourceImagePath?: string | null;
}

/**
 * An existing Library item (not in the Trash) that running these settings again would only repeat:
 * the same model family, prompt, size, seed, steps, CFG and - for a video - length and source image.
 * The newest one, or null. Prompts compare ignoring leading/trailing whitespace, as the form does.
 */
export function findDuplicateGeneration(db: DatabaseSync, modelFamily: string, query: DuplicateQuery): GenerationRecord | null {
  const isVideo = query.length !== undefined && query.length !== null;
  // A video's picture depends on its source image, which is only comparable once there is one.
  if (isVideo && !query.sourceImagePath) return null;
  const row = db
    .prepare(
      `${GENERATION_SELECT}
       WHERE g.model_family = ? AND g.trashed_at IS NULL AND TRIM(g.prompt) = ? AND g.width = ? AND g.height = ?
         AND g.seed = ? AND g.steps = ? AND ABS(g.cfg - ?) < 0.000001 AND g.length IS ?
         ${isVideo ? 'AND g.source_image_path = ?' : ''}
       ORDER BY g.id DESC LIMIT 1`
    )
    .get(
      modelFamily,
      query.prompt.trim(),
      query.width,
      query.height,
      query.seed,
      query.steps,
      query.cfg,
      isVideo ? (query.length as number) : null,
      ...(isVideo ? [query.sourceImagePath as string] : [])
    ) as unknown as GenerationRow | undefined;
  return row ? rowToRecord(row) : null;
}

/** Every pinned generation, most recently pinned first; hidden ones only when `showHidden`. */
export function listPinnedGenerations(db: DatabaseSync, showHidden: boolean): GenerationRecord[] {
  const rows = db
    .prepare(
      `${GENERATION_SELECT} WHERE g.pinned_at IS NOT NULL AND g.trashed_at IS NULL ${showHidden ? '' : 'AND g.hidden = 0'} ORDER BY g.pinned_at DESC, g.id DESC`
    )
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

/** Favorites that are in the Library. One in the Trash keeps its flag (so restoring brings it back as a
 * favorite) but its file belongs in the trash folder, so the startup sync must not touch it. */
export function listFavoriteIds(db: DatabaseSync): number[] {
  const rows = db.prepare('SELECT id FROM generations WHERE favorite = 1 AND trashed_at IS NULL').all() as unknown as { id: number }[];
  return rows.map((r) => r.id);
}
