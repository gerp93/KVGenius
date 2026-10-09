/**
 * What a saved snippet of wording is for. A "style" is a general look (anime, oil painting, "1930s movie poster") and a picture uses at most one.
 * An "element" is a reusable piece of the picture itself - a character's outfit, a recurring setting - and any number can be used together.
 */
export type StyleKind = 'style' | 'element';

export const STYLE_KINDS: readonly StyleKind[] = ['style', 'element'];

export const STYLE_KIND_LABEL: Record<StyleKind, string> = { style: 'Style', element: 'Element' };

/** A kind from outside (a form, a client), or 'style' for anything that is not one. */
export function cleanStyleKind(value: unknown): StyleKind {
  return value === 'element' ? 'element' : 'style';
}

/** A reusable snippet (e.g. "1930s movie poster, bold lithograph, limited palette") that Generate can add to a prompt, so the prompt box only
 * has to say what is in the picture. */
export interface PromptStyle {
  id: number;
  name: string;
  text: string;
  /** Style (a look, one per picture) or element (a reusable part of the picture, as many as wanted). */
  kind: StyleKind;
  createdAt: string;
}

/** What the Styles page sends when saving: a new style (no id) or an edit of an existing one. */
export interface PromptStyleInput {
  name: string;
  text: string;
  /** Absent means a style. */
  kind?: StyleKind;
}

export const MAX_STYLE_NAME_LENGTH = 60;
export const MAX_STYLE_TEXT_LENGTH = 2000;

/** A style's name and text as they will be stored (trimmed), or the reason they cannot be. */
export function validateStyleInput(input: PromptStyleInput): { ok: true; value: PromptStyleInput } | { ok: false; message: string } {
  const name = typeof input?.name === 'string' ? input.name.trim() : '';
  const text = typeof input?.text === 'string' ? input.text.trim() : '';
  if (!name) return { ok: false, message: 'Give it a name.' };
  if (name.length > MAX_STYLE_NAME_LENGTH) return { ok: false, message: `The name can be at most ${MAX_STYLE_NAME_LENGTH} characters.` };
  if (!text) return { ok: false, message: 'Enter the style text - the words to add to a prompt.' };
  if (text.length > MAX_STYLE_TEXT_LENGTH) return { ok: false, message: `The style text can be at most ${MAX_STYLE_TEXT_LENGTH} characters.` };
  return { ok: true, value: { name, text, kind: cleanStyleKind(input?.kind) } };
}

/**
 * The prompt that is actually sent: the user's own prompt followed by the extra wording - first the elements, then the style (a look comes
 * last, as it describes the whole picture). `extra` is one piece of wording or a list of them, in the order they go. With nothing (or only
 * blanks) the prompt is returned untouched - byte for byte what was typed - so everything downstream (the Library's stored prompt, Re-rack,
 * grouping by prompt) behaves exactly as it does without styles. Otherwise each piece is joined on with ", ", unless what comes before already
 * ends in punctuation, in which case a plain space is enough.
 */
export function combinePrompt(prompt: string, extra: string | null | undefined | readonly (string | null | undefined)[]): string {
  const pieces = (Array.isArray(extra) ? extra : [extra]).map((piece) => piece?.trim() ?? '').filter((piece) => piece !== '');
  if (pieces.length === 0) return prompt;
  let result = prompt.trimEnd();
  for (const piece of pieces) {
    if (!result.trim()) result = piece;
    else result = /[,;:.!?]$/.test(result) ? `${result} ${piece}` : `${result}, ${piece}`;
  }
  return result;
}

/** The wording to add to a prompt for the chosen elements (in the order they were picked) and style, in the order it goes. */
export function extraWording(elements: readonly PromptStyle[], style: PromptStyle | null): string[] {
  return [...elements.map((element) => element.text), ...(style ? [style.text] : [])];
}

/** What a generation records as its style (a label for the details panel only): the style and elements used, e.g. "Anime + Red jacket". */
export function styleLabel(elements: readonly PromptStyle[], style: PromptStyle | null): string | null {
  const names = [...(style ? [style.name] : []), ...elements.map((element) => element.name)];
  return names.length > 0 ? names.join(' + ') : null;
}
