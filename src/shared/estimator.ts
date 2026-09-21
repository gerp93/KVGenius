import { GenerationKind, TimeEstimate, TimingStatRow } from './types';

export interface EstimateQuery {
  family: string;
  kind: GenerationKind;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  /** Video frames (ignored for images). */
  lengthFrames: number | null;
  /** True when the previous run used the same model family, so its models are probably still loaded. */
  warm: boolean;
}

/** Newest runs considered; older ones reflect old hardware/driver/settings state. */
const MAX_HISTORY = 40;
/** Same-settings runs needed before trusting them over a scaled estimate. */
const MIN_EXACT = 3;
const DEFAULT_VIDEO_FRAMES = 81;

export function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * How much sampling work a run is. Time is close to proportional to it: image sampling scales with
 * pixels x steps (and roughly doubles when CFG > 1, which adds a second model pass); the video
 * template has a fixed number of steps so it scales with pixels x frames.
 */
function samplingUnit(q: Pick<EstimateQuery, 'kind' | 'width' | 'height' | 'steps' | 'cfg' | 'lengthFrames'>): number {
  const pixels = q.width * q.height;
  if (q.kind === 'video') return pixels * (q.lengthFrames ?? DEFAULT_VIDEO_FRAMES);
  return pixels * Math.max(1, q.steps) * (q.cfg > 1.05 ? 2 : 1);
}

/** How much decode/save work a run is: a still scales with pixels, a video with pixels x frames. */
function finishUnit(q: Pick<EstimateQuery, 'kind' | 'width' | 'height' | 'lengthFrames'>): number {
  const pixels = q.width * q.height;
  return q.kind === 'video' ? pixels * (q.lengthFrames ?? DEFAULT_VIDEO_FRAMES) : pixels;
}

function rowQuery(row: TimingStatRow): EstimateQuery {
  return {
    family: row.family,
    kind: row.kind,
    width: row.width,
    height: row.height,
    steps: row.steps,
    cfg: row.cfg,
    lengthFrames: row.length,
    warm: row.warm,
  };
}

function sameSettings(row: TimingStatRow, q: EstimateQuery): boolean {
  if (row.width !== q.width || row.height !== q.height) return false;
  if (q.kind === 'video') return row.length === q.lengthFrames;
  return row.steps === q.steps && Math.abs(row.cfg - q.cfg) < 0.01;
}

/**
 * Predicts a run's duration from earlier runs (newest first). Returns null until at least one run
 * of the same family has stage timings to learn from. Uses the runs' own timings when the same
 * settings were run enough times, and otherwise scales the typical seconds-per-unit-of-work to
 * the requested settings - so a size never run before still gets a sensible number.
 */
export function estimateRun(history: TimingStatRow[], q: EstimateQuery): TimeEstimate | null {
  const usable = history
    .filter(
      (r) =>
        r.family === q.family &&
        r.kind === q.kind &&
        r.samplingMs !== null &&
        r.finishMs !== null &&
        r.generateMs !== null
    )
    .slice(0, MAX_HISTORY);
  if (usable.length === 0) return null;

  const exact = usable.filter((r) => sameSettings(r, q));
  let samplingMs: number;
  let finishMs: number;
  let paceMs: number;
  let basis: TimeEstimate['basis'];
  let samples: number;

  if (exact.length >= MIN_EXACT) {
    basis = 'exact';
    samples = exact.length;
    samplingMs = median(exact.map((r) => r.samplingMs as number));
    finishMs = median(exact.map((r) => r.finishMs as number));
    const paces = exact.map((r) => r.paceMs).filter((p): p is number => p !== null);
    paceMs = paces.length ? median(paces) : samplingMs / Math.max(1, q.steps);
  } else {
    basis = 'scaled';
    samples = usable.length;
    const samplingRate = median(usable.map((r) => (r.samplingMs as number) / samplingUnit(rowQuery(r))));
    const finishRate = median(usable.map((r) => (r.finishMs as number) / finishUnit(rowQuery(r))));
    samplingMs = samplingRate * samplingUnit(q);
    finishMs = finishRate * finishUnit(q);
    const stepCounts = usable.map((r) => r.samplerSteps).filter((s): s is number => s !== null && s > 0);
    const steps = q.kind === 'image' ? Math.max(1, q.steps) : stepCounts.length ? median(stepCounts) : 1;
    paceMs = samplingMs / steps;
  }

  // Model loading depends on whether the models are probably already in memory, not on the settings.
  const loads = usable.filter((r) => r.warm === q.warm && r.loadMs !== null).map((r) => r.loadMs as number);
  const loadMs = loads.length ? median(loads) : null;

  const generateMs = samplingMs + finishMs;
  return { basis, samples, loadMs, samplingMs, finishMs, paceMs, generateMs, totalMs: generateMs + (loadMs ?? 0) };
}
