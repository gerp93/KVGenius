import { GenerationPhase, GenerationProgress } from '../shared/types';

/** One message from ComfyUI's websocket (`{ type, data }`). Binary preview frames never get here. */
export interface ComfyMessage {
  type: string;
  data?: { node?: string | null; prompt_id?: string; value?: number; max?: number; [key: string]: unknown };
}

/** How the time of a finished run divides up. All milliseconds; null when ComfyUI reported no steps. */
export interface RunTimings {
  totalMs: number;
  /** Time before sampling settled into its rhythm: loading models, encoding the prompt, GPU setup. */
  loadMs: number | null;
  /** All sampler steps, including the first (whose duration is inferred from the others). */
  samplingMs: number | null;
  /** Decoding and saving after the last step. */
  finishMs: number | null;
  samplerSteps: number | null;
  /** Typical time of one sampling step. */
  paceMs: number | null;
}

const isSampler = (cls: string | undefined) => !!cls && /sampler/i.test(cls);

/** Maps a ComfyUI node type to what the run is doing while that node executes. */
export function phaseForClass(cls: string | undefined): GenerationPhase {
  if (!cls) return 'working';
  if (/sampler/i.test(cls)) return 'preparing'; // becomes 'sampling' once a step reports
  if (/loader/i.test(cls)) return 'loading';
  if (/textencode/i.test(cls)) return 'encoding';
  if (/vaedecode/i.test(cls)) return 'decoding';
  if (/^save|createvideo|savevideo/i.test(cls)) return 'saving';
  return 'working';
}

/**
 * Follows one ComfyUI run from its websocket messages: which stage it is in, how far through
 * sampling, and - once it finishes - how the total time splits into loading / sampling / finishing.
 * Messages that arrive before the prompt id is known are held and replayed, so nothing is missed
 * between ComfyUI accepting the job and us learning its id.
 */
export class ProgressTracker {
  private promptId: string | null = null;
  private pending: ComfyMessage[] = [];
  private phase: GenerationPhase = 'starting';
  private stage = 0;
  private seenSamplers = new Set<string>();
  private samplers = new Map<string, { value: number; max: number }>();
  private firstStepAt: number | null = null;
  private lastStepAt: number | null = null;
  private currentStep = 0;
  private currentMax = 0;
  private doneAt: number | null = null;
  private stageCount: number;

  constructor(
    private readonly nodeClasses: Record<string, string>,
    private readonly startedAt: number,
    private readonly now: () => number = Date.now
  ) {
    this.stageCount = Object.values(nodeClasses).filter(isSampler).length;
  }

  setPromptId(promptId: string): GenerationProgress | null {
    this.promptId = promptId;
    let latest: GenerationProgress | null = null;
    for (const message of this.pending.splice(0)) latest = this.apply(message) ?? latest;
    return latest;
  }

  /** Feed one message; returns the new progress if it changed anything worth showing. */
  handle(message: ComfyMessage): GenerationProgress | null {
    if (this.promptId === null) {
      this.pending.push(message);
      return null;
    }
    return this.apply(message);
  }

  private apply(message: ComfyMessage): GenerationProgress | null {
    const data = message.data ?? {};
    if (data.prompt_id && this.promptId && data.prompt_id !== this.promptId) return null;

    if (message.type === 'execution_success' || (message.type === 'executing' && data.node === null)) {
      // ComfyUI tells us the moment it finishes; polling for the result can lag by up to a second.
      if (this.doneAt === null) this.doneAt = this.now();
      return null;
    }

    if (message.type === 'executing' && typeof data.node === 'string') {
      const cls = this.nodeClasses[data.node];
      if (isSampler(cls) && !this.seenSamplers.has(data.node)) {
        this.seenSamplers.add(data.node);
        this.stage = this.seenSamplers.size;
        this.currentStep = 0;
        this.currentMax = 0;
      }
      this.phase = phaseForClass(cls);
      return this.snapshot();
    }

    if (message.type === 'progress' && typeof data.value === 'number' && typeof data.max === 'number') {
      const node = typeof data.node === 'string' ? data.node : null;
      const cls = node ? this.nodeClasses[node] : undefined;
      // Only sampler progress counts as sampling; other nodes can report progress too.
      if (node && cls && !isSampler(cls)) return null;
      const key = node ?? 'sampler';
      if (!this.seenSamplers.has(key)) {
        this.seenSamplers.add(key);
        this.stage = this.seenSamplers.size;
      }
      this.samplers.set(key, { value: data.value, max: data.max });
      this.currentStep = data.value;
      this.currentMax = data.max;
      const at = this.now();
      if (this.firstStepAt === null) this.firstStepAt = at;
      this.lastStepAt = at;
      this.phase = 'sampling';
      return this.snapshot();
    }
    return null;
  }

  snapshot(): GenerationProgress {
    let stepsDone = 0;
    let stepsTotal = 0;
    for (const s of this.samplers.values()) {
      stepsDone += s.value;
      stepsTotal += s.max;
    }
    return {
      phase: this.phase,
      step: this.currentStep,
      stepMax: this.currentMax,
      stage: this.stage,
      stageCount: Math.max(this.stageCount, this.stage),
      stepsDone,
      // Only known once every sampler pass has reported its size.
      stepsTotal: this.samplers.size >= this.stageCount && this.samplers.size > 0 ? stepsTotal : null,
      firstStepAtMs: this.firstStepAt === null ? null : this.firstStepAt - this.startedAt,
      elapsedMs: this.now() - this.startedAt,
    };
  }

  /**
   * Splits the run's time. The first step's own duration is hidden inside the wait for the first
   * progress report, so it is taken from the pace of the others: loading = wait - one step.
   * The three parts always add up to the total.
   */
  finish(): RunTimings {
    const doneAt = this.doneAt ?? this.now();
    const totalMs = doneAt - this.startedAt;
    if (this.firstStepAt === null || this.lastStepAt === null) {
      return { totalMs, loadMs: null, samplingMs: null, finishMs: null, samplerSteps: null, paceMs: null };
    }
    let steps = 0;
    for (const s of this.samplers.values()) steps += s.max;
    const prepMs = this.firstStepAt - this.startedAt;
    const restMs = this.lastStepAt - this.firstStepAt;
    const finishMs = doneAt - this.lastStepAt;
    if (steps <= 1) {
      return { totalMs, loadMs: prepMs, samplingMs: 0, finishMs, samplerSteps: steps, paceMs: null };
    }
    const paceMs = restMs / (steps - 1);
    const stepInPrep = Math.min(paceMs, prepMs);
    return {
      totalMs,
      loadMs: prepMs - stepInPrep,
      samplingMs: restMs + stepInPrep,
      finishMs,
      samplerSteps: steps,
      paceMs,
    };
  }
}
