import type { GenerationParams } from './types';

/** Where a job came from: the app's own Generate page, or an outside client (MCP / local API). */
export type JobSource = 'ui' | 'mcp';

/**
 * queued -> running -> done | failed | cancelled. 'interrupted' is what a job that was queued or
 * running when the app closed becomes on the next start: its work is gone, and it is not re-run
 * on its own (the GPU should not start grinding through a stale backlog at launch).
 */
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';

export const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ['done', 'failed', 'cancelled', 'interrupted'];

export function isTerminalJobStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

/** What the caller supplies to queue a generation. */
export interface JobRequest {
  family: string;
  params: GenerationParams;
  source: JobSource;
  /** Groups jobs that belong together (e.g. the 15 clips of one music video). */
  batch?: string | null;
  /** The estimate that was shown for this run; stored with its actual timing. Not persisted on the job. */
  estimate?: { totalMs: number | null; generateMs: number | null } | null;
}

export interface JobInfo {
  id: number;
  source: JobSource;
  batch: string | null;
  family: string;
  params: GenerationParams;
  status: JobStatus;
  error: string | null;
  /** The Library record produced, once the job is done. */
  generationId: number | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface JobFilter {
  source?: JobSource;
  batch?: string;
  status?: JobStatus;
  limit?: number;
}
