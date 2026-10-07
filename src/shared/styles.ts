/** A reusable "how it looks" snippet (e.g. "1930s movie poster, bold lithograph, limited palette")
 * that Generate can append to a prompt, so the prompt box only has to say what is in the picture. */
export interface PromptStyle {
  id: number;
  name: string;
  text: string;
  createdAt: string;
}

/** What the Styles page sends when saving: a new style (no id) or an edit of an existing one. */
export interface PromptStyleInput {
  name: string;
  text: string;
}

export const MAX_STYLE_NAME_LENGTH = 60;
export const MAX_STYLE_TEXT_LENGTH = 2000;

/** A style's name and text as they will be stored (trimmed), or the reason they cannot be. */
export function validateStyleInput(input: PromptStyleInput): { ok: true; value: PromptStyleInput } | { ok: false; message: string } {
  const name = typeof input?.name === 'string' ? input.name.trim() : '';
  const text = typeof input?.text === 'string' ? input.text.trim() : '';
  if (!name) return { ok: false, message: 'Give the style a name.' };
  if (name.length > MAX_STYLE_NAME_LENGTH) return { ok: false, message: `The name can be at most ${MAX_STYLE_NAME_LENGTH} characters.` };
  if (!text) return { ok: false, message: 'Enter the style text - the words to add to a prompt.' };
  if (text.length > MAX_STYLE_TEXT_LENGTH) return { ok: false, message: `The style text can be at most ${MAX_STYLE_TEXT_LENGTH} characters.` };
  return { ok: true, value: { name, text } };
}

/**
 * The prompt that is actually sent: the user's own prompt followed by the style's words. With no style
 * (or a blank one) the prompt is returned untouched - byte for byte what was typed - so everything
 * downstream (the Library's stored prompt, Re-rack, grouping by prompt) behaves exactly as it does
 * without styles. Otherwise the two are joined with ", ", unless the prompt already ends in
 * punctuation, in which case a plain space is enough.
 */
export function combinePrompt(prompt: string, styleText: string | null | undefined): string {
  const style = styleText?.trim();
  if (!style) return prompt;
  const base = prompt.trimEnd();
  if (!base.trim()) return style;
  return /[,;:.!?]$/.test(base) ? `${base} ${style}` : `${base}, ${style}`;
}
