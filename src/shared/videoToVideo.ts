import { clampDenoise } from './imageToImage';
import { videoQualityFromCfg } from './videoQuality';

/**
 * Video to video: Wan 2.2's text-to-video pair of models, started from a supplied video instead of from empty noise - the video
 * counterpart of image to image. The source's frames are scaled to the output size (cropped from the centre if the shapes differ),
 * encoded to a latent, and only part of the way back to noise is sampled, so the clip keeps its motion and layout to the degree the
 * strength says and the prompt re-draws the rest. It is the text-to-video graph with the empty latent replaced, keeping every node
 * id, so model profiles (the text-to-video ones), the Fast / High switch and the sizes work unchanged.
 *
 * Only the first `length` frames are used, and the result plays at the source's frame rate, so a clip keeps its speed.
 */
export const V2V_FAMILY = 'wan22-v2v';

/** A starting strength: enough to restyle a clip while still following its motion. */
export const V2V_DEFAULT_STRENGTH = 0.7;

/** Sampling steps of the two quality choices and the step where the low-noise model takes over - the values of the primitives in
 * the template (`videoToVideo.test.ts` checks them against it). */
export const V2V_STEPS = {
  fast: { steps: 4, split: 2 },
  high: { steps: 20, split: 10 },
} as const;

export interface V2VSchedule {
  /** The step the first (high-noise) sampler starts at. */
  firstStart: number;
  /** The second sampler adds the noise itself, starting at `secondStart` - used when the strength is low enough that the high-noise stage is skipped. */
  secondAddsNoise: boolean;
  secondStart: number | null;
}

/**
 * Where in the schedule to begin so that about `strength` of the way back to noise is sampled (1 = from pure noise, as text to
 * video; lower keeps more of the source). Like ComfyUI's own denoise, the first `(1 - strength)` of the steps are skipped. The
 * high-noise sampler normally adds the noise and the low-noise one continues; when the start is already past the point where it
 * hands over, the low-noise sampler adds the noise itself.
 */
export function v2vSchedule(strength: unknown, cfg: number): V2VSchedule {
  const { steps, split } = V2V_STEPS[videoQualityFromCfg(cfg)];
  const s = clampDenoise(strength, V2V_DEFAULT_STRENGTH);
  const skip = Math.min(steps - 1, Math.max(0, Math.round(steps * (1 - s))));
  return skip < split ? { firstStart: skip, secondAddsNoise: false, secondStart: null } : { firstStart: split, secondAddsNoise: true, secondStart: skip };
}
