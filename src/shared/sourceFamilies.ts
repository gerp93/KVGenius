import { I2I_FAMILY, INPAINT_FAMILY, OUTPAINT_FAMILY } from './imageToImage';
import { UPSCALE_FAMILY } from './upscale';

/**
 * The model families that are made from a picture the person supplies: a video from its source image,
 * an upscale from the picture being enlarged. Each of their results keeps its own copy of that picture
 * (see main/sourceImages.ts), shows it as "Original" in the details panel, and can only be re-run while
 * the copy still exists.
 *
 * A new tool that works on a supplied picture belongs in this list - it is the one place the app asks.
 */
export const SOURCE_IMAGE_FAMILIES: ReadonlySet<string> = new Set(['wan22-i2v', I2I_FAMILY, INPAINT_FAMILY, OUTPAINT_FAMILY, UPSCALE_FAMILY]);

export function needsSourceImage(family: string): boolean {
  return SOURCE_IMAGE_FAMILIES.has(family);
}

/** Why a result cannot be re-run from its source: the copy it kept is gone. */
export const SOURCE_MISSING_MESSAGE = 'Not available - the source image is missing.';
