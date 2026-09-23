import { GenerationPhase, GenerationProgress, TimeEstimate } from '../../shared/types';

export interface RunDisplay {
  /** What the run is doing right now, in words. */
  label: string;
  /** False while the length of the wait is unknowable (loading models) - show a moving bar, no percentage. */
  determinate: boolean;
  /** 0..1, only meaningful when determinate. */
  fraction: number;
  /** Best guess at time left, or null when there is nothing to base one on. */
  remainingMs: number | null;
  /** True once the run has taken clearly longer than its estimate. */
  overrunning: boolean;
}

const LABELS: Record<GenerationPhase, string> = {
  starting: 'Starting...',
  loading: 'Loading models...',
  encoding: 'Reading the prompt...',
  preparing: 'Preparing the model on the GPU...',
  sampling: 'Generating',
  decoding: 'Decoding the result...',
  saving: 'Saving...',
  working: 'Working...',
};

/** Never let the bar reach 100% before the run really ends. */
const MAX_FRACTION = 0.97;

export interface ProgressInputs {
  progress: GenerationProgress | null;
  estimate: TimeEstimate | null;
  /** Milliseconds since the run started, by the local clock. */
  elapsedMs: number;
  /** Local-clock time (ms since the run started) of the last sampling step to report, or null. */
  lastStepAtMs: number | null;
}

/**
 * Turns ComfyUI's live stage/step reports and the run's estimate into what to show. ComfyUI's own
 * percentage counts workflow nodes, so it crawls and then leaps; this weights each stage by how
 * long it usually takes. Sampling is by far the most predictable part (each step takes about the
 * same), so that is where the percentage and the time left come from. Loading models has no
 * meaningful percentage, so it is shown as an animated bar.
 */
export function describeProgress({ progress, estimate, elapsedMs, lastStepAtMs }: ProgressInputs): RunDisplay {
  const phase = progress?.phase ?? 'starting';
  const total = progress?.stepsTotal ?? (progress && progress.stepMax > 0 ? progress.stepMax * progress.stageCount : null);
  const done = progress?.stepsDone ?? 0;

  let label = LABELS[phase];
  if (phase === 'sampling') {
    label = total ? `Generating - step ${Math.min(done, total)} of ${total}` : `Generating - step ${done}`;
    if (progress && progress.stageCount > 1) label += ` (pass ${progress.stage} of ${progress.stageCount})`;
  }

  const sampled = phase === 'sampling' || phase === 'decoding' || phase === 'saving';

  // Before the first step the wait is model loading / GPU setup: one-time pre-work with no
  // reliable duration of its own (a cold model can take anywhere from seconds to minutes,
  // depending on disk cache state that has nothing to do with the generation itself). It is
  // never flagged as "running long", and any time left shown here is against the load estimate
  // specifically - not the combined total, which would otherwise flag a normal cold load as
  // overrunning whenever there is no load history yet (load history defaults to unknown, not 0).
  if (!sampled || !total || total <= 0) {
    const remaining = estimate?.loadMs && elapsedMs < estimate.loadMs ? estimate.loadMs - elapsedMs : null;
    return { label, determinate: false, fraction: 0, remainingMs: remaining, overrunning: false };
  }

  // Once sampling has started, "running long" is judged only on the generating portion (time
  // since the first step) against the generate-only estimate - loading is excluded on both sides.
  const generateElapsedMs = progress?.firstStepAtMs != null ? Math.max(0, elapsedMs - progress.firstStepAtMs) : 0;
  const overrunning = !!estimate && generateElapsedMs > estimate.generateMs * 1.25 + 3000;

  const sinceStep = lastStepAtMs === null ? 0 : Math.max(0, elapsedMs - lastStepAtMs);
  const stepsLeft = Math.max(0, total - done);
  const finishMs = estimate?.finishMs ?? null;

  // Pace: measured from this run once a couple of steps are in, else what earlier runs showed.
  let pace: number | null = estimate?.paceMs ?? null;
  if (progress?.firstStepAtMs != null && done >= 2 && lastStepAtMs !== null) {
    pace = (lastStepAtMs - progress.firstStepAtMs) / (done - 1);
  }

  let fraction: number;
  let remaining: number | null = null;
  if (phase === 'sampling') {
    const samplingShare = total > 0 ? done / total : 0;
    if (estimate && estimate.generateMs > 0) {
      fraction = (samplingShare * estimate.samplingMs) / estimate.generateMs;
    } else {
      fraction = samplingShare * 0.95;
    }
    if (pace !== null) remaining = Math.max(0, stepsLeft * pace - sinceStep) + (finishMs ?? 0);
  } else {
    // Decoding / saving: sampling is done; creep through the last stretch over its usual duration.
    const samplingPart = estimate && estimate.generateMs > 0 ? estimate.samplingMs / estimate.generateMs : 0.95;
    const finishPart = 1 - samplingPart;
    const creep = finishMs && finishMs > 0 ? Math.min(1, sinceStep / finishMs) : 0.5;
    fraction = samplingPart + finishPart * creep;
    remaining = finishMs !== null ? Math.max(0, finishMs - sinceStep) : null;
  }
  return { label, determinate: true, fraction: Math.min(MAX_FRACTION, Math.max(0, fraction)), remainingMs: remaining, overrunning };
}
