/**
 * The two ways the wan22-i2v template can sample. "fast" uses the lightx2v 4-step LoRA (4 steps,
 * CFG 1) - quick but prone to grain; "high" turns the LoRA off and uses the model's own schedule
 * (20 steps, CFG 3.5 on the high-noise pass), which is cleaner but far slower.
 *
 * The choice is stored as ordinary steps/cfg on GenerationParams (and so in the library and the
 * timing stats), so runs made before this option existed - which always sent CFG 1 - read as "fast".
 */
export type VideoQuality = 'fast' | 'high';

export const VIDEO_QUALITY_SETTINGS: Record<VideoQuality, { steps: number; cfg: number }> = {
  fast: { steps: 4, cfg: 1 },
  high: { steps: 20, cfg: 3.5 },
};

export function videoQualityFromCfg(cfg: number): VideoQuality {
  return cfg > 1.05 ? 'high' : 'fast';
}

/** Rough sampling cost of "high" relative to "fast": 10 steps at CFG 3.5 (two model passes each)
 * plus 10 at CFG 1, against 4 steps at CFG 1. Only used to scale estimates for unseen settings. */
export const HIGH_VIDEO_WORK_FACTOR = 7.5;
