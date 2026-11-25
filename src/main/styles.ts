import { DatabaseSync } from 'node:sqlite';
import { PromptStyle, PromptStyleInput, cleanStyleKind, validateStyleInput } from '../shared/styles';
import { parseElementLora, serializeElementLora } from '../shared/loras';

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
  workspace_id INTEGER,
  created_at TEXT NOT NULL,
  text_changed_at TEXT NOT NULL DEFAULT '',
  lora TEXT
);
`;

/** A style is shared by every workspace (no workspace) or belongs to one; a name is unique among the styles one workspace can see
 * (checked in saveStyle) and within a workspace (this index, made once the column exists - see migrateSchema in db.ts). */
export const STYLES_NAME_INDEX = `
DROP INDEX IF EXISTS idx_styles_name;
CREATE UNIQUE INDEX IF NOT EXISTS idx_styles_scope_name ON styles (name COLLATE NOCASE, COALESCE(workspace_id, 0));
`;

interface StyleRow {
  id: number;
  name: string;
  text: string;
  kind?: string | null;
  workspace_id?: number | null;
  created_at: string;
  text_changed_at?: string | null;
  lora?: string | null;
}

function rowToStyle(row: StyleRow): PromptStyle {
  const kind = cleanStyleKind(row.kind);
  return {
    id: row.id,
    name: row.name,
    text: row.text,
    kind,
    workspaceId: row.workspace_id ?? null,
    createdAt: row.created_at,
    textChangedAt: row.text_changed_at || row.created_at,
    lora: kind === 'element' ? parseElementLora(row.lora) : null,
  };
}

function isNameTaken(err: unknown): boolean {
  return err instanceof Error && /UNIQUE constraint failed/i.test(err.message);
}

/** The styles a workspace can use: the shared ones and its own. Alphabetical (ignoring case), which is also the order of the Generate page's dropdown. */
export function listStyles(db: DatabaseSync, workspaceId: number): PromptStyle[] {
  return (
    db.prepare('SELECT * FROM styles WHERE workspace_id IS NULL OR workspace_id = ? ORDER BY name COLLATE NOCASE, id').all(workspaceId) as unknown as StyleRow[]
  ).map(rowToStyle);
}

export function getStyle(db: DatabaseSync, id: number): PromptStyle | null {
  const row = db.prepare('SELECT * FROM styles WHERE id = ?').get(id) as unknown as StyleRow | undefined;
  return row ? rowToStyle(row) : null;
}

/** A style or element by name among the ones a workspace can use. */
export function findStyleByName(db: DatabaseSync, name: string, workspaceId: number): PromptStyle | null {
  const row = db
    .prepare('SELECT * FROM styles WHERE name = ? COLLATE NOCASE AND (workspace_id IS NULL OR workspace_id = ?)')
    .get(name.trim(), workspaceId) as unknown as StyleRow | undefined;
  return row ? rowToStyle(row) : null;
}

/** Whether `name` is already used by something a style in this scope (null: shared by every workspace) would be confused with: a shared style
 * clashes with any style of that name, one for a workspace with a shared one or one of that workspace's own. */
function nameClashes(db: DatabaseSync, name: string, workspaceId: number | null, exceptId: number | null): boolean {
  const row =
    workspaceId === null
      ? db.prepare('SELECT id FROM styles WHERE name = ? COLLATE NOCASE AND id IS NOT ?').get(name, exceptId)
      : db.prepare('SELECT id FROM styles WHERE name = ? COLLATE NOCASE AND id IS NOT ? AND (workspace_id IS NULL OR workspace_id = ?)').get(name, exceptId, workspaceId);
  return row !== undefined;
}

/**
 * Saves a new style, or updates the one with `id`. A new one is for the current workspace (`workspaceId`) when the input says scope
 * 'workspace' - and, with no scope given, when it is an element - and otherwise shared by every workspace. Who can use an existing one is
 * not changed by editing it. Throws a message fit to show the user.
 */
export function saveStyle(db: DatabaseSync, input: PromptStyleInput, id: number | null, workspaceId: number): PromptStyle {
  const checked = validateStyleInput(input);
  if (!checked.ok) throw new Error(checked.message);
  const { name, text, kind } = checked.value;
  const now = new Date().toISOString();
  try {
    if (id === null) {
      const forWorkspace = checked.value.scope === undefined ? (kind ?? 'style') === 'element' : checked.value.scope === 'workspace';
      const owner = forWorkspace ? workspaceId : null;
      if (nameClashes(db, name, owner, null)) throw new Error(`A style or element named "${name}" already exists.`);
      const result = db
        .prepare('INSERT INTO styles (name, text, kind, workspace_id, created_at, text_changed_at, lora) VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run(name, text, kind ?? 'style', owner, now, now, serializeElementLora(checked.value.lora));
      return getStyle(db, Number(result.lastInsertRowid)) as PromptStyle;
    }
    const before = getStyle(db, id);
    if (before && nameClashes(db, name, before.workspaceId, id)) throw new Error(`A style or element named "${name}" already exists.`);
    const changedAt = before && before.text === text ? before.textChangedAt : now;
    // A LoRA that is not part of the input stays as it was - except that only an element can have one.
    const lora = checked.value.lora !== undefined ? checked.value.lora : kind === 'element' ? (before?.lora ?? null) : null;
    const result = db
      .prepare('UPDATE styles SET name = ?, text = ?, kind = ?, text_changed_at = ?, lora = ? WHERE id = ?')
      .run(name, text, kind ?? 'style', changedAt, serializeElementLora(lora), id);
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
