import { useCallback, useEffect, useRef, useState } from 'react';
import { GenerationParams, GenerationProgress, GenerationRecord, TimeEstimate } from '../../shared/types';
import { SlotViews, batchShownFor, shouldFollow } from '../../shared/slotViews';

export type JobKind = 'image' | 'video';
export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Job {
  id: number;
  /** Jobs queued by one Generate click (a batch of N, or a single one) share a batchId. */
  batchId: number;
  /** The Generate working tab that queued it, so its result shows in that tab only. None for jobs
   * from elsewhere (a Library upscale), which no tab's viewer shows. */
  slotId?: string;
  family: string;
  kind: JobKind;
  params: GenerationParams;
  status: JobStatus;
  /** Predicted duration from earlier runs; undefined = not asked yet, null = no history to go on. */
  estimate?: TimeEstimate | null;
  startedAt?: number;
  record?: GenerationRecord;
  imageUrl?: string;
  error?: string;
  /** A failed job the user has cleared from the queue panel (it stays a slot in the viewer). */
  dismissed?: boolean;
}

/** Live progress of the running job, plus when (local clock) the last sampling step reported. */
export interface ProgressInfo {
  progress: GenerationProgress;
  lastStepAt: number | null;
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

export type GenerationQueue = ReturnType<typeof useGenerationQueue>;

/**
 * Runs generations one at a time from a queue. ComfyUI only works on one prompt at once, so jobs
 * execute strictly in order; a failed job doesn't stop the ones behind it. Created once in App and
 * shared by the Generate page and the Library, so it keeps running while you browse other tabs.
 */
export function useGenerationQueue() {
  const [jobs, setJobs] = useState<Job[]>([]);
  // The batch each working tab is showing (see shared/slotViews.ts). The ref mirrors it so the
  // runner and enqueue can read the latest value without waiting for a render.
  const [views, setViews] = useState<SlotViews>({});
  const viewsRef = useRef<SlotViews>({});
  const [now, setNow] = useState(() => Date.now());
  const [progressInfo, setProgressInfo] = useState<ProgressInfo | null>(null);
  // Bumped whenever a run finishes: the estimates for what is still waiting can use its timings.
  const [historyVersion, setHistoryVersion] = useState(0);

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

  const pointView = useCallback((slotId: string, batchId: number) => {
    viewsRef.current = { ...viewsRef.current, [slotId]: batchId };
    setViews(viewsRef.current);
  }, []);

  const patchJob = useCallback((id: number, patch: Partial<Job>) => {
    setJobs((prev) => prev.map((j) => (j.id === id ? { ...j, ...patch } : j)));
  }, []);

  // Live stage/step updates from ComfyUI for the running job.
  useEffect(
    () =>
      window.kvgenius.onGenerationProgress((progress) => {
        const at = Date.now();
        setProgressInfo((prev) => ({
          progress,
          // When a sampling step last completed - used to time the stretch since it.
          lastStepAt:
            progress.phase === 'sampling' && (!prev || prev.progress.stepsDone !== progress.stepsDone || prev.lastStepAt === null)
              ? at
              : (prev?.lastStepAt ?? null),
        }));
      }),
    []
  );

  // Runner: whenever nothing is running and something is queued, start the next job.
  useEffect(() => {
    if (runningRef.current) return;
    const next = jobs.find((j) => j.status === 'queued');
    if (!next) return;

    runningRef.current = true;
    cancelRequestedRef.current = false;
    patchJob(next.id, { status: 'running', startedAt: Date.now() });
    setProgressInfo(null);
    // If the batch its tab is viewing has nothing left to do, follow the queue onto this job's batch.
    if (next.slotId !== undefined && shouldFollow(jobsRef.current, viewsRef.current, next.slotId)) {
      pointView(next.slotId, next.batchId);
    }

    // The estimate on screen for this job is what gets stored next to its actual time; if none has
    // arrived yet (a job started at once), ask for one first so the run is not left without.
    const start = async () => {
      let estimate = next.estimate;
      if (estimate === undefined) {
        try {
          estimate = await window.kvgenius.estimateGeneration(next.family, next.params);
        } catch {
          estimate = null;
        }
      }
      return window.kvgenius.generate(
        next.family,
        next.params,
        estimate ? { totalMs: estimate.totalMs, generateMs: estimate.generateMs } : null
      );
    };

    start().then(
      (result) => {
        runningRef.current = false;
        setProgressInfo(null);
        setHistoryVersion((v) => v + 1);
        setJobs((prev) => {
          const updated = prev.map((j) =>
            j.id === next.id ? { ...j, status: 'done' as const, record: result.record, imageUrl: result.imageUrl } : j
          );
          return updated.length > MAX_HISTORY ? updated.filter((j, i) => isActive(j) || i >= updated.length - MAX_HISTORY) : updated;
        });
        // A finished result shows in the tab it was made for - never in the others.
        if (next.slotId !== undefined) pointView(next.slotId, next.batchId);
      },
      (err) => {
        runningRef.current = false;
        setProgressInfo(null);
        if (cancelRequestedRef.current) {
          // Cancelled on purpose: the job just disappears.
          setJobs((prev) => prev.filter((j) => j.id !== next.id));
        } else {
          patchJob(next.id, { status: 'failed', error: cleanError(err) });
        }
      }
    );
  }, [jobs, patchJob, pointView]);

  // Keep every waiting/running job's estimate current. A job's estimate depends on the family that
  // runs just before it (same family = models already loaded), so each is asked about in queue order.
  const activeKey = jobs
    .filter(isActive)
    .map((j) => `${j.id}:${j.status}`)
    .join(',');
  useEffect(() => {
    const active = jobsRef.current.filter(isActive);
    if (active.length === 0) return;
    let cancelled = false;
    active.forEach((job, index) => {
      // The first job follows whatever ran last (the main process knows); later ones follow the job ahead.
      const before = index === 0 ? undefined : active[index - 1].family;
      window.kvgenius
        .estimateGeneration(job.family, job.params, before)
        .catch(() => null)
        .then((estimate) => {
          if (!cancelled) patchJob(job.id, { estimate });
        });
    });
    return () => {
      cancelled = true;
    };
  }, [activeKey, historyVersion, patchJob]);

  /** Adds jobs as one batch, queued from working tab `slotId` (none for jobs made elsewhere).
   * Returns how many fit under MAX_PENDING_JOBS. */
  const enqueue = useCallback((items: NewJob[], slotId?: string): number => {
    const pending = jobsRef.current.filter(isActive).length;
    const accepted = items.slice(0, Math.max(0, MAX_PENDING_JOBS - pending));
    if (accepted.length === 0) return 0;

    const batchId = nextBatchRef.current++;
    const created: Job[] = accepted.map((item) => ({
      id: nextIdRef.current++,
      batchId,
      slotId,
      family: item.family,
      kind: item.kind,
      params: item.params,
      status: 'queued',
    }));
    setJobs((prev) => [...prev, ...created]);
    // Show the new batch straight away unless its tab is still looking at one being worked on.
    if (slotId !== undefined && shouldFollow(jobsRef.current, viewsRef.current, slotId)) pointView(slotId, batchId);
    return accepted.length;
  }, [pointView]);

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

  /** Puts an existing record on screen in working tab `slotId` (e.g. recalled from the Library) as
   * a finished one-off. */
  const showRecord = useCallback((record: GenerationRecord, kind: JobKind, imageUrl: string, slotId: string) => {
    const batchId = nextBatchRef.current++;
    const job: Job = {
      id: nextIdRef.current++,
      batchId,
      slotId,
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
    pointView(slotId, batchId);
  }, [pointView]);

  /** A working tab was closed: it no longer has a view to keep. */
  const forgetSlot = useCallback((slotId: string) => {
    const { [slotId]: _gone, ...rest } = viewsRef.current;
    viewsRef.current = rest;
    setViews(rest);
  }, []);

  /** A generation was deleted: drop its result from the viewer. */
  const removeRecord = useCallback((recordId: number) => {
    setJobs((prev) => prev.filter((j) => j.record?.id !== recordId));
  }, []);

  const updateRecord = useCallback((recordId: number, patch: Partial<GenerationRecord>) => {
    setJobs((prev) =>
      prev.map((j) => (j.record?.id === recordId ? { ...j, record: { ...j.record, ...patch } as GenerationRecord } : j))
    );
  }, []);

  /** A file was moved on disk (favorited / unfavorited): repoint everything that referred to the
   * old path - the finished result's record and URL, and any job that uses it as a source image. */
  const relocateFile = useCallback((recordId: number, oldPath: string, newPath: string, newUrl: string) => {
    if (oldPath === newPath) return;
    setJobs((prev) =>
      prev.map((j) => {
        let next = j;
        if (next.record?.id === recordId) {
          next = { ...next, record: { ...next.record, imagePath: newPath }, imageUrl: newUrl };
        }
        if (next.params.sourceImagePath === oldPath) {
          next = { ...next, params: { ...next.params, sourceImagePath: newPath } };
        }
        return next;
      })
    );
  }, []);

  /** The batch working tab `slotId` should show (null: it has no results yet). If every job of its
   * viewed batch was cancelled away it falls back to that tab's newest batch. */
  const viewBatchFor = useCallback((slotId: string) => batchShownFor(jobs, views, slotId), [jobs, views]);

  return {
    jobs,
    now,
    progressInfo,
    viewBatchFor,
    forgetSlot,
    enqueue,
    cancelJob,
    clearQueued,
    dismissFailed,
    showRecord,
    updateRecord,
    removeRecord,
    relocateFile,
  };
}
