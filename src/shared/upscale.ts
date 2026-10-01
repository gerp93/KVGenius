/** Model family id of the image-upscale template. Deliberately not in FAMILY_KIND: that map also
 * decides which families outside clients may generate with, and an upscale needs a source image. */
export const UPSCALE_FAMILY = 'upscale-image';

/** Output sizes offered, as a multiple of the source image's size. Upscale models enlarge by a
 * fixed amount (usually 4x); the template then resizes the result to exactly this factor. */
export const UPSCALE_FACTORS = [1.5, 2, 3, 4];
export const DEFAULT_UPSCALE_FACTOR = 2;
