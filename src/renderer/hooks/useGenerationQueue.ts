import { useCallback, useEffect, useRef, useState } from 'react';
import { GenerationParams, GenerationRecord } from '../../shared/types';

export type JobKind = 'image' | 'video';
export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Job {
  id: number;
  /** Jobs queued by one Generate click (a batch of N, or a single one) share a batchId. */
  batchId: number;
  family: string;
  kind: JobKind;
  params: GenerationParams;
  status: JobStatus;
  startedAt?: number;
  record?: GenerationRecord;
  imageUrl?: string;
  error?: string;
  /** A failed job the user has cleared from the queue panel (it stays a slot in the viewer). */
  dismissed?: boolean;
}

export interface NewJob {
  family: string;
  kind: JobKind;
  params: GenerationParams;
}

/** Most jobs that can be waiting at once - a guard against queueing hundreds by accident. */
export const MAX_PENDING_JOBS = 50;
/** Most jobs one Generate click can add (each gets its own random seed). */
export const MAX_BATCH_SIZE = 10;

const MAX_HISTORY = 300;

function isActive(job: Job): boolean {
  return job.status === 'queued' || job.status === 'running';
}

function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  // Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method".
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
}

/**
 * Runs generations one at a time from a queue. ComfyUI only works on one prompt at once, so jobs
 * execute strictly in order; a failed job doesn't stop the ones behind it. Lives in the Generate
 * page, which stays mounted while you browse other tabs, so the queue keeps running.
 */
export function useGenerationQueue() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [viewBatchId, setViewBatchId] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const jobsRef = useRef<Job[]>([]);
  const nextIdRef = useRef(1);
  const nextBatchRef = useRef(1);
  const runningRef = useRef(false);
  const cancelRequestedRef = useRef(false);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  const hasRunning = jobs.some((j) => j.status === 'running');
  useEffect(() => {
    if (!hasRunning) return;
    setNow(Date.now());
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [hasRunning]);

  const patchJob = useCallback((id: number, patch: Partial<Job>) => {
    setJobs((prev) => prev.map((j) => (j.id === id ? { ...j, ...patch } : j)));
  }, []);

  // Runner: whenever nothing is running and something is queued, start the next job.
  useEffect(() => {
    if (runningRef.current) return;
    const next = jobs.find((j) => j.status === 'queued');
    if (!next) return;

    runningRef.current = true;
    cancelRequestedRef.current = false;
    patchJob(next.id, { status: 'running', startedAt: Date.now() });
    // If the batch being viewed has nothing left to do, follow the queue onto this job's batch.
    setViewBatchId((current) =>
      current === null || !jobsRef.current.some((j) => j.batchId === current && isActive(j)) ? next.batchId : current
    );

    window.kvgenius.generate(next.family, next.params).then(
      (result) => {
        runningRef.current = false;
        setJobs((prev) => {
          const updated = prev.map((j) =>
            j.id === next.id ? { ...j, status: 'done' as const, record: result.record, imageUrl: result.imageUrl } : j
          );
          return updated.length > MAX_HISTORY ? updated.filter((j, i) => isActive(j) || i >= updated.length - MAX_HISTORY) : updated;
        });
        setViewBatchId(next.batchId);
      },
      (err) => {
        runningRef.current = false;
        if (cancelRequestedRef.current) {
          // Cancelled on purpose: the job just disappears.
          setJobs((prev) => prev.filter((j) => j.id !== next.id));
        } else {
          patchJob(next.id, { status: 'failed', error: cleanError(err) });
        }
      }
    );
  }, [jobs, patchJob]);

  /** Adds jobs as one batch. Returns how many fit under MAX_PENDING_JOBS. */
  const enqueue = useCallback((items: NewJob[]): number => {
    const pending = jobsRef.current.filter(isActive).length;
    const accepted = items.slice(0, Math.max(0, MAX_PENDING_JOBS - pending));
    if (accepted.length === 0) return 0;

    const batchId = nextBatchRef.current++;
    const created: Job[] = accepted.map((item) => ({
      id: nextIdRef.current++,
      batchId,
      family: item.family,
      kind: item.kind,
      params: item.params,
      status: 'queued',
    }));
    setJobs((prev) => [...prev, ...created]);
    // Starting from idle: show the new batch straight away. While busy, leave the viewer alone.
    if (pending === 0) setViewBatchId(batchId);
    return accepted.length;
  }, []);

  /** Cancels a running job (interrupting ComfyUI) or removes a waiting one. */
  const cancelJob = useCallback((id: number) => {
    const job = jobsRef.current.find((j) => j.id === id);
    if (!job) return;
    if (job.status === 'running') {
      cancelRequestedRef.current = true;
      void window.kvgenius.cancelGeneration();
    } else if (job.status === 'queued') {
      setJobs((prev) => prev.filter((j) => j.id !== id));
    }
  }, []);

  const clearQueued = useCallback(() => {
    setJobs((prev) => prev.filter((j) => j.status !== 'queued'));
  }, []);

  const dismissFailed = useCallback((id: number) => patchJob(id, { dismissed: true }), [patchJob]);

  /** Puts an existing record on screen (e.g. recalled from the Library) as a finished one-off. */
  const showRecord = useCallback((record: GenerationRecord, kind: JobKind, imageUrl: string) => {
    const batchId = nextBatchRef.current++;
    const job: Job = {
      id: nextIdRef.current++,
      batchId,
      family: record.modelFamily,
      kind,
      params: {
        prompt: record.prompt,
        width: record.width,
        height: record.height,
        seed: record.seed,
        steps: record.steps,
        cfg: record.cfg,
      },
      status: 'done',
      record,
      imageUrl,
    };
    setJobs((prev) => [...prev, job]);
    setViewBatchId(batchId);
  }, []);

  const updateRecord = useCallback((recordId: number, patch: Partial<GenerationRecord>) => {
    setJobs((prev) =>
      prev.map((j) => (j.record?.id === recordId ? { ...j, record: { ...j.record, ...patch } as GenerationRecord } : j))
    );
  }, []);

  // If every job of the viewed batch was cancelled away, fall back to the latest batch that still
  // has something to show instead of leaving the viewer empty.
  const shownBatchId = jobs.some((j) => j.batchId === viewBatchId)
    ? viewBatchId
    : jobs.length > 0
      ? jobs[jobs.length - 1].batchId
      : null;

  return {
    jobs,
    now,
    viewBatchId: shownBatchId,
    enqueue,
    cancelJob,
    clearQueued,
    dismissFailed,
    showRecord,
    updateRecord,
  };
}
