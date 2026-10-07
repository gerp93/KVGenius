import { GIF_FAMILY } from './gif';
import { isUpscaleFamily } from './upscale';

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
