/**
 * Image to image: the same Z-Image model and graph as text to image, but the sampler starts from a picture
 * (encoded to a latent) instead of from empty noise, and only re-draws part of it. How much is the
 * "strength" - ComfyUI's `denoise`: 1 ignores the picture completely, a small number keeps nearly all of it.
 * It is its own family key because its workflow differs and it is made from a supplied picture
 * (see shared/sourceFamilies.ts); model profiles still apply, since the loader nodes are the same.
 */
export const I2I_FAMILY = 'z-image-i2i';

/**
 * Inpainting: image to image with a painted mask, so only the masked spots are re-drawn and the rest is the original.
 * Its own family (its workflow adds the mask steps and pastes the result back over the original), made from two supplied
 * pictures - the source image and the mask - and using the same Z-Image model files, so profiles still apply.
 */
export const INPAINT_FAMILY = 'z-image-inpaint';

/**
 * Outpainting: expands a picture beyond its frame. The source image is padded on any side (grey, with a mask marking the new
 * area), only the new area is drawn - from the prompt, continuing what is at the edge - and the result is pasted back over the
 * original, so the original's pixels are untouched. Its own family (its workflow adds the padding), made from one supplied
 * picture, using the same Z-Image model files, so profiles still apply.
 */
export const OUTPAINT_FAMILY = 'z-image-outpaint';

/** How far a side can be extended, in pixels of the source picture. */
export const OUTPAINT_MAX_PAD = 2048;

/** How much of the new area is re-drawn. It starts from a blurred stretch of the picture (so it already has the picture's colours and layout at
 * its edges) and is only partly re-drawn: at 1 the model ignores that start and draws the area from the prompt alone, which is what made the new
 * area look like an unrelated picture. */
export const OUTPAINT_DEFAULT_DENOISE = 0.8;

export interface OutpaintPadding {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export const OUTPAINT_SIDES = ['left', 'top', 'right', 'bottom'] as const;

/** A padding from outside (a form, a config, a client): whole pixels within limits, or null if nothing is extended. */
export function normalizeOutpaint(value: unknown): OutpaintPadding | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  const out = { left: 0, top: 0, right: 0, bottom: 0 };
  for (const side of OUTPAINT_SIDES) {
    const n = Number(v[side]);
    out[side] = Number.isFinite(n) ? Math.min(OUTPAINT_MAX_PAD, Math.max(0, Math.round(n))) : 0;
  }
  return OUTPAINT_SIDES.some((side) => out[side] > 0) ? out : null;
}

/** The padding as one stable string (the database column and the duplicate guard compare it), or null for none. */
export function serializeOutpaint(value: OutpaintPadding | null | undefined): string | null {
  const pad = normalizeOutpaint(value);
  return pad ? `${pad.left},${pad.top},${pad.right},${pad.bottom}` : null;
}

/** Reads `serializeOutpaint`'s string back; anything unreadable is no padding. */
export function parseOutpaint(text: string | null | undefined): OutpaintPadding | null {
  if (!text) return null;
  const [left, top, right, bottom] = text.split(',').map(Number);
  return normalizeOutpaint({ left, top, right, bottom });
}

/** The size the extended picture is drawn at: its whole canvas (source plus padding), the long side kept between 1024 and
 * 1536 (so a small picture is not drawn tiny and a big one not enormous), both sides a multiple of 64. */
export function outpaintOutputSize(sourceWidth: number, sourceHeight: number, pad: OutpaintPadding): { width: number; height: number } {
  const totalWidth = sourceWidth + pad.left + pad.right;
  const totalHeight = sourceHeight + pad.top + pad.bottom;
  const longSide = Math.min(1536, Math.max(1024, Math.max(totalWidth, totalHeight)));
  const scale = longSide / Math.max(totalWidth, totalHeight);
  const snap = (n: number) => Math.min(2048, Math.max(256, Math.round((n * scale) / 64) * 64));
  return { width: snap(totalWidth), height: snap(totalHeight) };
}

export const DENOISE_LIMITS = { min: 0.05, max: 1 } as const;

/** A starting point that visibly changes a picture without throwing it away; the user sets what suits. */
export const DEFAULT_DENOISE = 0.6;

/** Keeps a strength within what the sampler accepts, falling back to the default for anything that is not a number. */
export function clampDenoise(value: unknown, fallback: number = DEFAULT_DENOISE): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(DENOISE_LIMITS.max, Math.max(DENOISE_LIMITS.min, Math.round(n * 100) / 100));
}

/** The workflow family a Generate run uses: outpainting once the picture is extended, inpainting once a mask is painted on
 * the source image, image to image with just a source image, else text to image. A mask or an extension without a source
 * image means nothing, so it is ignored. */
export function imageFamilyFor(textFamily: string, hasStartPicture: boolean, hasMask = false, hasOutpaint = false): string {
  if (!hasStartPicture) return textFamily;
  if (hasOutpaint) return OUTPAINT_FAMILY;
  return hasMask ? INPAINT_FAMILY : I2I_FAMILY;
}

/** Whether a record's family is one of those that start from a supplied picture (with or without a mask or padding). */
export function isPictureStartFamily(family: string): boolean {
  return family === I2I_FAMILY || family === INPAINT_FAMILY || family === OUTPAINT_FAMILY;
}
