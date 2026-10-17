import { DatabaseSync } from 'node:sqlite';
import { PromptTemplate, PromptTemplateInput, TemplateSection, cleanTemplateSeparator, validateTemplateInput } from '../shared/promptTemplates';

/** User-defined prompt templates (see shared/promptTemplates.ts). Names are unique ignoring case; the sections are stored as JSON. */
export const PROMPT_TEMPLATES_SCHEMA = `
CREATE TABLE IF NOT EXISTS prompt_templates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL COLLATE NOCASE,
  sections TEXT NOT NULL,
  separator TEXT NOT NULL DEFAULT 'comma',
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_templates_name ON prompt_templates (name COLLATE NOCASE);
`;

interface TemplateRow {
  id: number;
  name: string;
  sections: string;
  separator: string | null;
  created_at: string;
}

function parseSections(json: string): TemplateSection[] {
  try {
    const parsed: unknown = JSON.parse(json);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((s): s is Record<string, unknown> => !!s && typeof s === 'object')
      .map((s) => ({ id: String(s.id ?? ''), label: String(s.label ?? ''), hint: String(s.hint ?? '') }))
      .filter((s) => s.id !== '' && s.label !== '');
  } catch {
    return [];
  }
}

function rowToTemplate(row: TemplateRow): PromptTemplate {
  return { id: row.id, name: row.name, sections: parseSections(row.sections), separator: cleanTemplateSeparator(row.separator), createdAt: row.created_at };
}

/** Alphabetical (ignoring case), which is also the order of the Generate page's dropdown. */
export function listPromptTemplates(db: DatabaseSync): PromptTemplate[] {
  return (db.prepare('SELECT * FROM prompt_templates ORDER BY name COLLATE NOCASE, id').all() as unknown as TemplateRow[]).map(rowToTemplate);
}

export function getPromptTemplate(db: DatabaseSync, id: number): PromptTemplate | null {
  const row = db.prepare('SELECT * FROM prompt_templates WHERE id = ?').get(id) as unknown as TemplateRow | undefined;
  return row ? rowToTemplate(row) : null;
}

/** Saves a new template, or updates the one with `id`. Throws a message fit to show the user. */
export function savePromptTemplate(db: DatabaseSync, input: PromptTemplateInput, id: number | null = null): PromptTemplate {
  const checked = validateTemplateInput(input);
  if (!checked.ok) throw new Error(checked.message);
  const { name, sections, separator } = checked.value;
  try {
    if (id === null) {
      const result = db
        .prepare('INSERT INTO prompt_templates (name, sections, separator, created_at) VALUES (?, ?, ?, ?)')
        .run(name, JSON.stringify(sections), separator, new Date().toISOString());
      return getPromptTemplate(db, Number(result.lastInsertRowid)) as PromptTemplate;
    }
    const result = db.prepare('UPDATE prompt_templates SET name = ?, sections = ?, separator = ? WHERE id = ?').run(name, JSON.stringify(sections), separator, id);
    if (Number(result.changes) === 0) throw new Error('That template no longer exists.');
    return getPromptTemplate(db, id) as PromptTemplate;
  } catch (err) {
    if (err instanceof Error && /UNIQUE constraint failed/i.test(err.message)) throw new Error(`A template named "${name}" already exists.`);
    throw err;
  }
}

/** Deleting a template never touches a prompt written with it: a tab that used it just goes back to the single prompt box. */
export function deletePromptTemplate(db: DatabaseSync, id: number): void {
  db.prepare('DELETE FROM prompt_templates WHERE id = ?').run(id);
}
