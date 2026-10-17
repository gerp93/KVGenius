import { combinePrompt } from './styles';

/**
 * A prompt template is only a way of writing a prompt: a named list of sections (outfit, camera, style...) that the Generate page shows as
 * separate text boxes in place of the one big prompt box. The boxes are joined back into a single ordinary prompt, so nothing downstream
 * (the queue, the Library, Re-rack) knows or cares that a template was used.
 */
export type TemplateSeparator = 'comma' | 'space' | 'newline';

export const TEMPLATE_SEPARATORS: readonly TemplateSeparator[] = ['comma', 'space', 'newline'];

export const TEMPLATE_SEPARATOR_LABEL: Record<TemplateSeparator, string> = {
  comma: 'Commas (a, b, c)',
  space: 'Spaces (a b c)',
  newline: 'New lines',
};

/** A separator from outside (a form, a stored row), or the default for anything that is not one. */
export function cleanTemplateSeparator(value: unknown): TemplateSeparator {
  return value === 'space' || value === 'newline' ? value : 'comma';
}

export interface TemplateSection {
  /** Stable across renames and reordering, so what was typed into a section stays with it when the template is edited. */
  id: string;
  /** The box's heading, e.g. "Outfit". */
  label: string;
  /** Greyed-out text shown in the empty box; optional. */
  hint: string;
}

export interface PromptTemplate {
  id: number;
  name: string;
  sections: TemplateSection[];
  separator: TemplateSeparator;
  createdAt: string;
}

/** What the Templates page sends when saving: a new template (no id) or an edit of an existing one. */
export interface PromptTemplateInput {
  name: string;
  sections: { id?: string; label: string; hint?: string }[];
  /** Absent means commas. */
  separator?: TemplateSeparator;
}

export const MAX_TEMPLATE_NAME_LENGTH = 60;
export const MAX_TEMPLATE_SECTIONS = 12;
export const MAX_SECTION_LABEL_LENGTH = 40;
export const MAX_SECTION_HINT_LENGTH = 120;

/** What was typed into each section of the active template, by section id. */
export type TemplateValues = Record<string, string>;

/** A new, unused section id for a template being edited. */
export function newSectionId(taken: readonly string[] = []): string {
  let id: string;
  do {
    id = `s${Math.random().toString(36).slice(2, 8)}`;
  } while (taken.includes(id));
  return id;
}

/** A template's name and sections as they will be stored (trimmed, every section with a unique id), or the reason they cannot be. */
export function validateTemplateInput(input: PromptTemplateInput): { ok: true; value: { name: string; sections: TemplateSection[]; separator: TemplateSeparator } } | { ok: false; message: string } {
  const name = typeof input?.name === 'string' ? input.name.trim() : '';
  if (!name) return { ok: false, message: 'Give the template a name.' };
  if (name.length > MAX_TEMPLATE_NAME_LENGTH) return { ok: false, message: `The name can be at most ${MAX_TEMPLATE_NAME_LENGTH} characters.` };
  const raw = Array.isArray(input?.sections) ? input.sections : [];
  if (raw.length === 0) return { ok: false, message: 'Add at least one section.' };
  if (raw.length > MAX_TEMPLATE_SECTIONS) return { ok: false, message: `A template can have at most ${MAX_TEMPLATE_SECTIONS} sections.` };
  const sections: TemplateSection[] = [];
  for (const section of raw) {
    const label = typeof section?.label === 'string' ? section.label.trim() : '';
    const hint = typeof section?.hint === 'string' ? section.hint.trim() : '';
    if (!label) return { ok: false, message: 'Every section needs a name.' };
    if (label.length > MAX_SECTION_LABEL_LENGTH) return { ok: false, message: `A section name can be at most ${MAX_SECTION_LABEL_LENGTH} characters.` };
    if (hint.length > MAX_SECTION_HINT_LENGTH) return { ok: false, message: `A section hint can be at most ${MAX_SECTION_HINT_LENGTH} characters.` };
    const wanted = typeof section.id === 'string' && section.id.trim() !== '' ? section.id.trim() : '';
    const id = wanted !== '' && !sections.some((s) => s.id === wanted) ? wanted : newSectionId(sections.map((s) => s.id));
    sections.push({ id, label, hint });
  }
  return { ok: true, value: { name, sections, separator: cleanTemplateSeparator(input?.separator) } };
}

/** The single prompt a template's boxes make: the non-blank ones in order, joined as the template says. Blank boxes leave no trace. */
export function joinSections(template: Pick<PromptTemplate, 'sections' | 'separator'>, values: TemplateValues): string {
  const pieces = template.sections.map((section) => (values[section.id] ?? '').trim()).filter((piece) => piece !== '');
  if (template.separator === 'space') return pieces.join(' ');
  if (template.separator === 'newline') return pieces.join('\n');
  // combinePrompt joins on ", " unless a piece already ends in punctuation.
  return combinePrompt('', pieces);
}

/** The values for a template just picked: whatever was already written in the single prompt box goes into the first section, so nothing is lost. */
export function valuesFromPrompt(template: Pick<PromptTemplate, 'sections'>, prompt: string): TemplateValues {
  const first = template.sections[0];
  return first && prompt.trim() !== '' ? { [first.id]: prompt } : {};
}
