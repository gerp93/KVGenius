import { DatabaseSync } from 'node:sqlite';
import { GenerationProgress } from '../shared/types';
import { JobFilter, JobInfo, JobRequest, JobSource, isTerminalJobStatus } from '../shared/jobs';
import {
  countQueuedJobs,
  finishJob,
  getJob,
  insertJob,
  listJobs,
  markInterruptedJobs,
  markJobRunning,
  nextQueuedJob,
} from './jobStore';

/** Most jobs that may be waiting at once - a guard against a client queueing thousands by accident. */
export const MAX_QUEUED_JOBS = 200;

export interface JobRunContext {
  estimate: JobRequest['estimate'];
  onProgress: (progress: GenerationProgress) => void;
}

/** Does the actual work for one job and returns the Library record it produced. */
export type JobRunner = (job: JobInfo, context: JobRunContext) => Promise<{ generationId: number }>;

export interface JobQueueOptions {
  /** Asks the backend to stop whatever is running right now (ComfyUI interrupt + local abort). */
  cancelRunning: () => Promise<void>;
  /** True for the error a runner throws when it was stopped by cancelRunning(). */
  isCancellation?: (err: unknown) => boolean;
}

type JobListener = (job: JobInfo) => void;
type ProgressListener = (jobId: number, progress: GenerationProgress) => void;

/**
 * Runs generation jobs strictly one at a time (ComfyUI only works on one prompt at once), in the
 * order they were submitted, whatever their source. The queue lives in the main process so the
 * app's own UI and outside clients share one line for the GPU and one view of what is happening.
 * A failed job never stops the ones behind it.
 */
export class JobQueue {
  private runningId: number | null = null;
  private pumping = false;
  private readonly cancelRequested = new Set<number>();
  private readonly estimates = new Map<number, JobRequest['estimate']>();
  private readonly waiters = new Map<number, Array<(job: JobInfo) => void>>();
  private readonly jobListeners = new Set<JobListener>();
  private readonly progressListeners = new Set<ProgressListener>();

  constructor(
    private readonly db: DatabaseSync,
    private readonly runner: JobRunner,
    private readonly options: JobQueueOptions
  ) {
    // Whatever was waiting or running when the app last closed can never finish now.
    markInterruptedJobs(db);
  }

  /** Queues a job and returns it (status 'queued'). Throws if the queue is full. */
  submit(request: JobRequest): JobInfo {
    if (countQueuedJobs(this.db) >= MAX_QUEUED_JOBS) {
      throw new Error(`The queue is full (${MAX_QUEUED_JOBS} jobs waiting). Wait for some to finish or cancel some.`);
    }
    const job = insertJob(this.db, request);
    this.estimates.set(job.id, request.estimate ?? null);
    this.emitJob(job);
    void this.pump();
    return job;
  }

  get(id: number): JobInfo | null {
    return getJob(this.db, id);
  }

  list(filter?: JobFilter): JobInfo[] {
    return listJobs(this.db, filter);
  }

  /** Resolves with the job once it reaches a final state (done, failed, cancelled or interrupted). */
  wait(id: number): Promise<JobInfo> {
    const job = getJob(this.db, id);
    if (!job) return Promise.reject(new Error(`No job ${id}.`));
    if (isTerminalJobStatus(job.status)) return Promise.resolve(job);
    return new Promise((resolve) => {
      const list = this.waiters.get(id) ?? [];
      list.push(resolve);
      this.waiters.set(id, list);
    });
  }

  /**
   * Cancels a waiting job (it never runs) or the running one (ComfyUI is interrupted). Returns
   * false when there is nothing to cancel: unknown id, or the job already finished.
   */
  async cancel(id: number): Promise<boolean> {
    const job = getJob(this.db, id);
    if (!job || isTerminalJobStatus(job.status)) return false;
    if (job.status === 'queued') {
      this.settle(id, 'cancelled', { error: null });
      return true;
    }
    this.cancelRequested.add(id);
    await this.options.cancelRunning();
    return true;
  }

  /** Cancels whichever job is running right now, if any. */
  async cancelRunning(): Promise<boolean> {
    return this.runningId === null ? false : this.cancel(this.runningId);
  }

  /** Cancels every job still waiting (not the running one), optionally only those of one batch
   * and/or one source. Returns how many. */
  cancelQueued(batch?: string, source?: JobSource): number {
    const waiting = listJobs(this.db, { status: 'queued', batch, source, limit: 500 });
    for (const job of waiting) this.settle(job.id, 'cancelled', { error: null });
    return waiting.length;
  }

  onJobChanged(listener: JobListener): () => void {
    this.jobListeners.add(listener);
    return () => this.jobListeners.delete(listener);
  }

  onProgress(listener: ProgressListener): () => void {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      for (;;) {
        const job = nextQueuedJob(this.db);
        if (!job) return;
        await this.run(job);
      }
    } finally {
      this.pumping = false;
    }
  }

  private async run(job: JobInfo): Promise<void> {
    this.runningId = job.id;
    markJobRunning(this.db, job.id);
    this.emitJob(getJob(this.db, job.id) as JobInfo);
    try {
      const { generationId } = await this.runner(job, {
        estimate: this.estimates.get(job.id) ?? null,
        onProgress: (progress) => this.progressListeners.forEach((l) => l(job.id, progress)),
      });
      this.settle(job.id, 'done', { generationId });
    } catch (err) {
      const stopped = this.cancelRequested.has(job.id) || (this.options.isCancellation?.(err) ?? false);
      if (stopped) this.settle(job.id, 'cancelled', { error: null });
      else this.settle(job.id, 'failed', { error: err instanceof Error ? err.message : String(err) });
    } finally {
      this.runningId = null;
      this.cancelRequested.delete(job.id);
      this.estimates.delete(job.id);
    }
  }

  private settle(
    id: number,
    status: 'done' | 'failed' | 'cancelled',
    detail: { error?: string | null; generationId?: number | null }
  ): void {
    finishJob(this.db, id, status, detail);
    this.estimates.delete(id);
    const job = getJob(this.db, id) as JobInfo;
    this.emitJob(job);
    const waiting = this.waiters.get(id);
    if (waiting) {
      this.waiters.delete(id);
      waiting.forEach((resolve) => resolve(job));
    }
  }

  private emitJob(job: JobInfo): void {
    this.jobListeners.forEach((l) => l(job));
  }
}
