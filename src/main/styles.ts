import { DatabaseSync } from 'node:sqlite';
import { PromptStyle, PromptStyleInput, cleanStyleKind, validateStyleInput } from '../shared/styles';

/**
 * User-defined prompt styles (see shared/styles.ts). Names are unique ignoring case, so a client can
 * ask for a style by name (the MCP `style` argument) without ambiguity.
 */
export const STYLES_SCHEMA = `
CREATE TABLE IF NOT EXISTS styles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL COLLATE NOCASE,
  text TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'style',
  created_at TEXT NOT NULL,
  text_changed_at TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_styles_name ON styles (name COLLATE NOCASE);
`;

interface StyleRow {
  id: number;
  name: string;
  text: string;
  kind?: string | null;
  created_at: string;
  text_changed_at?: string | null;
}

function rowToStyle(row: StyleRow): PromptStyle {
  return { id: row.id, name: row.name, text: row.text, kind: cleanStyleKind(row.kind), createdAt: row.created_at, textChangedAt: row.text_changed_at || row.created_at };
}

function isNameTaken(err: unknown): boolean {
  return err instanceof Error && /UNIQUE constraint failed/i.test(err.message);
}

/** Alphabetical (ignoring case), which is also the order of the Generate page's dropdown. */
export function listStyles(db: DatabaseSync): PromptStyle[] {
  return (db.prepare('SELECT * FROM styles ORDER BY name COLLATE NOCASE, id').all() as unknown as StyleRow[]).map(rowToStyle);
}

export function getStyle(db: DatabaseSync, id: number): PromptStyle | null {
  const row = db.prepare('SELECT * FROM styles WHERE id = ?').get(id) as unknown as StyleRow | undefined;
  return row ? rowToStyle(row) : null;
}

export function findStyleByName(db: DatabaseSync, name: string): PromptStyle | null {
  const row = db.prepare('SELECT * FROM styles WHERE name = ? COLLATE NOCASE').get(name.trim()) as unknown as StyleRow | undefined;
  return row ? rowToStyle(row) : null;
}

/** Saves a new style, or updates the one with `id`. Throws a message fit to show the user. */
export function saveStyle(db: DatabaseSync, input: PromptStyleInput, id: number | null = null): PromptStyle {
  const checked = validateStyleInput(input);
  if (!checked.ok) throw new Error(checked.message);
  const { name, text, kind } = checked.value;
  const now = new Date().toISOString();
  try {
    if (id === null) {
      const result = db
        .prepare('INSERT INTO styles (name, text, kind, created_at, text_changed_at) VALUES (?, ?, ?, ?, ?)')
        .run(name, text, kind ?? 'style', now, now);
      return getStyle(db, Number(result.lastInsertRowid)) as PromptStyle;
    }
    const before = getStyle(db, id);
    const changedAt = before && before.text === text ? before.textChangedAt : now;
    const result = db.prepare('UPDATE styles SET name = ?, text = ?, kind = ?, text_changed_at = ? WHERE id = ?').run(name, text, kind ?? 'style', changedAt, id);
    if (Number(result.changes) === 0) throw new Error('That style no longer exists.');
    return getStyle(db, id) as PromptStyle;
  } catch (err) {
    if (isNameTaken(err)) throw new Error(`A style or element named "${name}" already exists.`);
    throw err;
  }
}

/** Deleting a style never touches past generations: they keep the text they were made with. */
export function deleteStyle(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM styles WHERE id = ?').run(id);
}
