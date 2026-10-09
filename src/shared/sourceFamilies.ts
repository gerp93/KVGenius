import { I2I_FAMILY, INPAINT_FAMILY, OUTPAINT_FAMILY } from './imageToImage';
import { UPSCALE_FAMILY } from './upscale';
import { V2V_FAMILY } from './videoToVideo';

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

/** The same for a video to video result, which is re-drawn from a Library video read where it is (there is no kept copy). */
export const SOURCE_VIDEO_MISSING_MESSAGE = 'Not available - the source video is missing (it was deleted or moved).';

/** The message for a result whose source is gone. */
export function sourceMissingMessage(family: string): string {
  return family === V2V_FAMILY ? SOURCE_VIDEO_MISSING_MESSAGE : SOURCE_MISSING_MESSAGE;
}
