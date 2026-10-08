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

export const DENOISE_LIMITS = { min: 0.05, max: 1 } as const;

/** A starting point that visibly changes a picture without throwing it away; the user sets what suits. */
export const DEFAULT_DENOISE = 0.6;

/** Keeps a strength within what the sampler accepts, falling back to the default for anything that is not a number. */
export function clampDenoise(value: unknown): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return DEFAULT_DENOISE;
  return Math.min(DENOISE_LIMITS.max, Math.max(DENOISE_LIMITS.min, Math.round(n * 100) / 100));
}

/** The workflow family a Generate run uses: inpainting once a mask is painted on the source image, image to image with
 * just a source image, else text to image. A mask without a source image means nothing, so it is ignored. */
export function imageFamilyFor(textFamily: string, hasStartPicture: boolean, hasMask = false): string {
  if (!hasStartPicture) return textFamily;
  return hasMask ? INPAINT_FAMILY : I2I_FAMILY;
}

/** Whether a record's family is one of the two that start from a supplied picture (with or without a mask). */
export function isPictureStartFamily(family: string): boolean {
  return family === I2I_FAMILY || family === INPAINT_FAMILY;
}
