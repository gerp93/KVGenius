/**
 * LoRAs on image generation. A LoRA is a small add-on file in ComfyUI's `models/loras` folder that teaches a base model something it
 * does not know (a species, a character, a look). KVGenius carries them on *elements* (see shared/styles.ts): an element's wording
 * holds the trigger words, and its optional LoRA says which file to load, how strongly, and for which model families.
 *
 * Only some families can load one so far (LORA_FAMILIES); an element's LoRA used with any other family is left out of the run and the
 * user is told, never an error. Adding a family later means an entry in LORA_FAMILIES / LORA_FAMILY_LABEL here and one in
 * LORA_NODE_MAP (main/loraPatch.ts).
 */

/** What a job carries for one LoRA: the file name ComfyUI lists, and how strongly to apply it. */
export interface LoraSpec {
  file: string;
  strength: number;
}

/** A LoRA saved on an element: the spec plus the families it is meant for. */
export interface ElementLora extends LoraSpec {
  families: string[];
}

/** The families whose workflow can load a LoRA. */
export const LORA_FAMILIES: readonly string[] = ['flux-dev'];

/** How the user knows each of those families. */
export const LORA_FAMILY_LABEL: Record<string, string> = { 'flux-dev': 'Flux dev' };

/** The most LoRAs one picture chains (each is one more model load step). */
export const MAX_LORAS = 3;

export const LORA_STRENGTH = { min: -2, max: 2, default: 1 } as const;

export const MAX_LORA_FILE_LENGTH = 300;

export function loraFamilySupported(family: string): boolean {
  return LORA_FAMILIES.includes(family);
}

/** "Flux dev" (or "Flux dev or X" for several) - for hints and notes. */
export function loraFamilyNames(families: readonly string[] = LORA_FAMILIES): string {
  const names = families.map((f) => LORA_FAMILY_LABEL[f] ?? f);
  return names.length <= 1 ? (names[0] ?? '') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

function cleanFamilies(value: unknown): string[] {
  const list = Array.isArray(value) ? value.filter((f): f is string => typeof f === 'string' && LORA_FAMILIES.includes(f)) : [];
  const unique = [...new Set(list)];
  return unique.length > 0 ? unique : [...LORA_FAMILIES];
}

/**
 * An element's LoRA as typed in the form, checked: `null` for none (no file given), or the cleaned value, or the reason it cannot be
 * saved. The strength defaults to 1; the families default to every family that can load one.
 */
export function validateElementLora(input: unknown): { ok: true; value: ElementLora | null } | { ok: false; message: string } {
  if (input === null || input === undefined) return { ok: true, value: null };
  if (typeof input !== 'object') return { ok: false, message: 'The LoRA is not valid.' };
  const raw = input as { file?: unknown; strength?: unknown; families?: unknown };
  const file = typeof raw.file === 'string' ? raw.file.trim() : '';
  if (!file) return { ok: true, value: null };
  if (file.length > MAX_LORA_FILE_LENGTH) return { ok: false, message: `The LoRA file name can be at most ${MAX_LORA_FILE_LENGTH} characters.` };
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(file)) return { ok: false, message: 'The LoRA file name has characters that cannot be in a file name.' };
  const strength = raw.strength === undefined || raw.strength === null || raw.strength === '' ? LORA_STRENGTH.default : Number(raw.strength);
  if (!Number.isFinite(strength) || strength < LORA_STRENGTH.min || strength > LORA_STRENGTH.max) {
    return { ok: false, message: `The LoRA strength must be a number from ${LORA_STRENGTH.min} to ${LORA_STRENGTH.max}.` };
  }
  return { ok: true, value: { file, strength, families: cleanFamilies(raw.families) } };
}

/** How an element's LoRA is stored in the styles table (JSON), or null for none. */
export function serializeElementLora(lora: ElementLora | null | undefined): string | null {
  return lora ? JSON.stringify({ file: lora.file, strength: lora.strength, families: lora.families }) : null;
}

/** Reads a stored element LoRA back; anything unreadable counts as none. */
export function parseElementLora(raw: string | null | undefined): ElementLora | null {
  if (!raw) return null;
  try {
    const checked = validateElementLora(JSON.parse(raw));
    return checked.ok ? checked.value : null;
  } catch {
    return null;
  }
}

/** The LoRAs of a job as stored on a Library record (JSON), or null for none - also what the duplicate guard compares. */
export function serializeLoras(loras: readonly LoraSpec[] | null | undefined): string | null {
  return loras && loras.length > 0 ? JSON.stringify(loras.map((l) => ({ file: l.file, strength: l.strength }))) : null;
}

/** Reads stored job LoRAs back; null for none or anything unreadable. */
export function parseLoras(raw: string | null | undefined): LoraSpec[] | null {
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const list = parsed.flatMap((entry) => {
      const e = entry as { file?: unknown; strength?: unknown } | null;
      return e && typeof e.file === 'string' && e.file !== '' && typeof e.strength === 'number' && Number.isFinite(e.strength) ? [{ file: e.file, strength: e.strength }] : [];
    });
    return list.length > 0 ? list : null;
  } catch {
    return null;
  }
}

/** One short line for a LoRA, e.g. "ffl-twi.safetensors x 0.8". */
export function describeLora(lora: LoraSpec): string {
  return `${lora.file} × ${lora.strength}`;
}

/** The part of an element that matters here (see PromptStyle). */
interface ElementWithLora {
  name: string;
  lora?: ElementLora | null;
}

export interface LoraResolution {
  /** What the job carries: the LoRAs that apply to this family, in the order the elements were picked. */
  applied: LoraSpec[];
  /** Notes for what was left out, fit to show the user. */
  notes: string[];
}

/**
 * Which of the chosen elements' LoRAs a picture of `family` uses. An element without one adds nothing. A LoRA is left out - with a note,
 * never an error - when this family cannot load one, when the element says it is for other families, when the same file is already in
 * use (it would be applied twice over), or when more than MAX_LORAS are asked for.
 */
export function resolveLoras(elements: readonly ElementWithLora[], family: string): LoraResolution {
  const applied: LoraSpec[] = [];
  const notes: string[] = [];
  for (const element of elements) {
    const lora = element.lora;
    if (!lora) continue;
    if (!loraFamilySupported(family) || !lora.families.includes(family)) {
      const forFamilies = loraFamilyNames(lora.families.filter(loraFamilySupported));
      notes.push(`"${element.name}" has a LoRA (${describeLora(lora)}) that only works with ${forFamilies || 'some other model'}, so it is not used for this picture - its words are still added.`);
    } else if (applied.some((a) => a.file === lora.file)) {
      notes.push(`"${element.name}" uses ${lora.file}, which another element already loads - it is applied once.`);
    } else if (applied.length >= MAX_LORAS) {
      notes.push(`Only ${MAX_LORAS} LoRAs can be used at once - the one for "${element.name}" (${lora.file}) is left out.`);
    } else {
      applied.push({ file: lora.file, strength: lora.strength });
    }
  }
  return { applied, notes };
}
