/**
 * Image to image: the same Z-Image model and graph as text to image, but the sampler starts from a picture
 * (encoded to a latent) instead of from empty noise, and only re-draws part of it. How much is the
 * "strength" - ComfyUI's `denoise`: 1 ignores the picture completely, a small number keeps nearly all of it.
 * It is its own family key because its workflow differs and it is made from a supplied picture
 * (see shared/sourceFamilies.ts); model profiles still apply, since the loader nodes are the same.
 */
export const I2I_FAMILY = 'z-image-i2i';

export const DENOISE_LIMITS = { min: 0.05, max: 1 } as const;

/** A starting point that visibly changes a picture without throwing it away; the user sets what suits. */
export const DEFAULT_DENOISE = 0.6;

/** Keeps a strength within what the sampler accepts, falling back to the default for anything that is not a number. */
export function clampDenoise(value: unknown): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return DEFAULT_DENOISE;
  return Math.min(DENOISE_LIMITS.max, Math.max(DENOISE_LIMITS.min, Math.round(n * 100) / 100));
}

/** The workflow family a Generate run uses: image to image once a start picture is chosen, else text to image. */
export function imageFamilyFor(textFamily: string, hasStartPicture: boolean): string {
  return hasStartPicture ? I2I_FAMILY : textFamily;
}
