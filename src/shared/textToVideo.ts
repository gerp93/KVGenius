/**
 * Text to video: Wan 2.2's text-to-video pair of models, run from a prompt alone. It shares the image-to-video
 * workflow's shape (two-stage high/low noise sampling, the same text encoder and VAE, a Fast 4-step LoRA switch) with
 * the start picture replaced by an empty video latent - so the node ids match `wan22-i2v`'s and the Fast / High
 * quality choice works unchanged. Unlike image to video it is made from no supplied picture, so it is not in
 * shared/sourceFamilies.ts.
 */
export const I2V_FAMILY = 'wan22-i2v';
export const T2V_FAMILY = 'wan22-t2v';

/** Whether a video of this family can be extended: only what the app made from a prompt or a picture (not an upscale, whose size is not the
 * clip's, nor a GIF). */
export function isExtendableFamily(family: string): boolean {
  return family === I2V_FAMILY || family === T2V_FAMILY;
}

/** The video workflow family a Generate run uses: image to video once the run starts from a picture, else text to video. */
export function videoFamilyFor(fromPicture: boolean): string {
  return fromPicture ? I2V_FAMILY : T2V_FAMILY;
}
