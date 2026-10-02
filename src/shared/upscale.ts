/** Model family ids of the upscale templates. Outside clients may not generate with these (an upscale
 * needs an existing library file), so apiService rejects them via isUpscaleFamily(). The image one is
 * left out of FAMILY_KIND (unknown families count as images); the video one is in it so its output is
 * filed with the videos. */
export const UPSCALE_FAMILY = 'upscale-image';
export const UPSCALE_VIDEO_FAMILY = 'upscale-video';

export function isUpscaleFamily(family: string): boolean {
  return family === UPSCALE_FAMILY || family === UPSCALE_VIDEO_FAMILY;
}

/** Output sizes offered, as a multiple of the source image's size. Upscale models enlarge by a
 * fixed amount (usually 4x); the template then resizes the result to exactly this factor. */
export const UPSCALE_FACTORS = [1.5, 2, 3, 4];
export const DEFAULT_UPSCALE_FACTOR = 2;
