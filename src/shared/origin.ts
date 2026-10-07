import { GIF_FAMILY } from './gif';
import { FAMILY_KIND, GenerationKind } from './types';
import { UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY, isUpscaleFamily } from './upscale';

/** How a Library item came to be - shown as a small tag on its card so it can be told at a glance. */
export type OriginKind = 'text-to-image' | 'image-to-video' | 'upscale' | 'gif';

export interface GenerationOrigin {
  kind: OriginKind;
  /** The short tag text. */
  label: string;
  /** The longer explanation, for the tag's tooltip. */
  title: string;
}

const ORIGINS: Record<OriginKind, Omit<GenerationOrigin, 'kind'>> = {
  'text-to-image': { label: 'Text → Image', title: 'Generated from a text prompt' },
  'image-to-video': { label: 'Image → Video', title: 'A video animated from a still image' },
  upscale: { label: 'Upscaled', title: 'An enlarged copy of another picture or video, not generated from the prompt' },
  gif: { label: 'GIF', title: 'A GIF made from a video' },
};

/** The origin of an item made by this model family (a `GenerationRecord.modelFamily`), or null for a
 * family this does not know - better no tag than a wrong one. */
export function generationOrigin(modelFamily: string): GenerationOrigin | null {
  let kind: OriginKind | null = null;
  if (isUpscaleFamily(modelFamily)) kind = 'upscale';
  else if (modelFamily === GIF_FAMILY) kind = 'gif';
  else if (modelFamily === 'z-image-turbo') kind = 'text-to-image';
  else if (modelFamily === 'wan22-i2v') kind = 'image-to-video';
  return kind ? { kind, ...ORIGINS[kind] } : null;
}

/** Every origin, in the order the Library's filter lists them. */
export const ORIGIN_KINDS: OriginKind[] = ['text-to-image', 'image-to-video', 'upscale', 'gif'];

/** The model families that make an item of this origin - what a "show only these" filter selects. */
export const FAMILIES_BY_ORIGIN: Record<OriginKind, string[]> = {
  'text-to-image': ['z-image-turbo'],
  'image-to-video': ['wan22-i2v'],
  upscale: [UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY],
  gif: [GIF_FAMILY],
};

/** The label of an origin, for the filter's buttons. */
export function originLabel(kind: OriginKind): string {
  return ORIGINS[kind].label;
}

/** Keeps only real origins (a list from outside, such as the renderer, may hold anything), without repeats. */
export function cleanOrigins(value: unknown): OriginKind[] {
  if (!Array.isArray(value)) return [];
  return ORIGIN_KINDS.filter((kind) => value.includes(kind));
}

/** Whether an item made by this family passes an origin filter (no origins selected: everything does). */
export function passesOriginFilter(modelFamily: string, origins: readonly OriginKind[]): boolean {
  if (origins.length === 0) return true;
  const origin = generationOrigin(modelFamily);
  return origin !== null && origins.includes(origin.kind);
}

/** Which Library tab an item of this model family is filed under (unknown families count as images). */
function kindOfFamily(modelFamily: string): GenerationKind {
  return FAMILY_KIND[modelFamily] === 'video' ? 'video' : 'image';
}

/** The origins that can occur on a Library tab - what its filter offers. Upscaled is on both, since
 * pictures and videos can both be upscaled; Text -> Image and GIF are only on Images, Image -> Video
 * only on Videos, so a chip is never offered where it could not match anything. */
export function originsForKind(kind: GenerationKind): OriginKind[] {
  return ORIGIN_KINDS.filter((origin) => FAMILIES_BY_ORIGIN[origin].some((family) => kindOfFamily(family) === kind));
}
