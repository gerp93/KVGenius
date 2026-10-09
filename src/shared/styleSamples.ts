/**
 * The Styles page's examples: one picture per style or element, all made from the same standard prompt and the
 * same seed, plus one with no wording added (the baseline), so what each snippet does is the only thing that differs.
 */

export const DEFAULT_SAMPLE_PROMPT = 'a young woman standing in a city street, looking at the camera';
/** A fixed seed: the point of the examples is that only the wording changes between them. */
export const DEFAULT_SAMPLE_SEED = 20240611;
export const MAX_SAMPLE_PROMPT_LENGTH = 1000;
export const MAX_SAMPLE_SEED = 2 ** 32 - 1;

/** Examples are small square pictures: quick to make, and the same shape for every card. */
export const SAMPLE_SIZE = 768;
export const SAMPLE_STEPS = 8;
export const SAMPLE_CFG = 1;

/** The id the baseline (no style, no element) is stored and requested under. */
export const BASELINE_ID = 0;

export interface StyleSampleSettings {
  /** What every example is made from, before the style's or element's own wording is added. */
  prompt: string;
  seed: number;
}

export const DEFAULT_SAMPLE_SETTINGS: StyleSampleSettings = { prompt: DEFAULT_SAMPLE_PROMPT, seed: DEFAULT_SAMPLE_SEED };

/** Settings from outside (a form), checked; a message fit to show when they cannot be used. */
export function validateSampleSettings(input: { prompt?: unknown; seed?: unknown }): { ok: true; value: StyleSampleSettings } | { ok: false; message: string } {
  const prompt = typeof input.prompt === 'string' ? input.prompt.trim() : '';
  if (prompt === '') return { ok: false, message: 'Write the standard prompt the examples are made from.' };
  if (prompt.length > MAX_SAMPLE_PROMPT_LENGTH) return { ok: false, message: `The standard prompt can be at most ${MAX_SAMPLE_PROMPT_LENGTH} characters.` };
  const seed = typeof input.seed === 'number' ? input.seed : Number(input.seed);
  if (!Number.isInteger(seed) || seed < 0 || seed > MAX_SAMPLE_SEED) return { ok: false, message: `The seed must be a whole number from 0 to ${MAX_SAMPLE_SEED}.` };
  return { ok: true, value: { prompt, seed } };
}

/** What an example was made from, kept with it so a later change can be told apart. */
export interface SampleSnapshot {
  /** The style's or element's wording when it was rendered ('' for the baseline). */
  text: string;
  prompt: string;
  seed: number;
}

/** 'current': made from the wording and standard prompt as they are now. 'outdated': one of them changed since. 'none': no picture to show. */
export type SampleFreshness = 'none' | 'current' | 'outdated';

export function sampleFreshness(snapshot: SampleSnapshot | null, text: string, settings: StyleSampleSettings): SampleFreshness {
  if (!snapshot) return 'none';
  return snapshot.text === text && snapshot.prompt === settings.prompt && snapshot.seed === settings.seed ? 'current' : 'outdated';
}

/** 'queued' / 'rendering': a job for it is waiting / running now. 'failed': the last attempt did not produce a picture. */
export type SampleState = SampleFreshness | 'queued' | 'rendering' | 'failed';

export interface StyleSampleInfo {
  /** The style's or element's id, or BASELINE_ID. */
  id: number;
  state: SampleState;
  /** The picture, when there is one (it stays shown, marked outdated, while a new one is made). */
  imageUrl: string | null;
  renderedAt: string | null;
  /** Why the last attempt failed. */
  error: string | null;
}

export interface StyleSamplesView {
  settings: StyleSampleSettings;
  samples: StyleSampleInfo[];
}

/** The ids whose example should be made again: no picture, outdated, or (with `all`) every one. Failed ones are retried. */
export function idsToRender(samples: readonly StyleSampleInfo[], all: boolean): number[] {
  return samples
    .filter((s) => s.state !== 'queued' && s.state !== 'rendering' && (all || s.state !== 'current'))
    .map((s) => s.id);
}
