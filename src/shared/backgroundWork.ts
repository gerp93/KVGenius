import type { JobInfo } from './jobs';

/** The queue batch label of every Styles-page example job, so they are recognisable in the queue. */
export const STYLE_SAMPLES_BATCH = 'style-examples';

/**
 * Jobs in the main-process queue that the window did not queue itself - the Styles page's examples and requests
 * from outside clients (MCP). They use the same GPU, in the same line, but are not listed in the queue panel; the
 * panel only says that they are holding the queue up.
 */
export interface BackgroundWork {
  /** Style / element examples waiting or being made (the one being made included). */
  styleExamples: number;
  /** Requests from outside clients waiting or being made. */
  outside: number;
  /** What the job being made right now is, when it is one of these - the queue is waiting on it. */
  running: 'styleExample' | 'outside' | null;
}

export const NO_BACKGROUND_WORK: BackgroundWork = { styleExamples: 0, outside: 0, running: null };

type BackgroundJob = Pick<JobInfo, 'batch' | 'source' | 'status'>;

/** Whether a job is one the window does not show in its queue. */
export function isBackgroundJob(job: Pick<JobInfo, 'batch' | 'source'>): boolean {
  return job.batch === STYLE_SAMPLES_BATCH || job.source === 'mcp';
}

/** Counts the waiting and running background jobs among `jobs`. */
export function summarizeBackgroundWork(jobs: readonly BackgroundJob[]): BackgroundWork {
  const work: BackgroundWork = { styleExamples: 0, outside: 0, running: null };
  for (const job of jobs) {
    if ((job.status !== 'queued' && job.status !== 'running') || !isBackgroundJob(job)) continue;
    const isExample = job.batch === STYLE_SAMPLES_BATCH;
    if (isExample) work.styleExamples++;
    else work.outside++;
    if (job.status === 'running') work.running = isExample ? 'styleExample' : 'outside';
  }
  return work;
}

export function sameBackgroundWork(a: BackgroundWork, b: BackgroundWork): boolean {
  return a.styleExamples === b.styleExamples && a.outside === b.outside && a.running === b.running;
}

/**
 * What the queue panel says when the window has pictures of its own to make, or null when the background work is not in the
 * way. Examples run last, so only one already being made holds anything up; outside requests are in the same line as the
 * window's own jobs.
 */
export function describeBackgroundWork(work: BackgroundWork): string | null {
  if (work.running === 'styleExample') return 'Waiting for a style example that was already being made';
  const outside = `${work.outside} request${work.outside === 1 ? '' : 's'} from outside the app`;
  if (work.running === 'outside') return `Waiting for ${outside}`;
  if (work.outside > 0) return `${outside} also in the queue`;
  return null;
}
