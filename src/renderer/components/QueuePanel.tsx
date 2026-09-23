import { GenerationRecord } from '../../shared/types';
import { Job, JobKind, ProgressInfo } from '../hooks/useGenerationQueue';
import { formatDuration } from '../utils/format';
import { describeProgress } from '../utils/progressModel';
import { framesToSeconds } from '../utils/video';
import GeneratedVideo from './GeneratedVideo';
import RunProgress from './RunProgress';

interface Props {
  jobs: Job[];
  now: number;
  progressInfo: ProgressInfo | null;
  collapsed: boolean;
  onToggle: () => void;
  onCancelJob: (id: number) => void;
  onClearQueued: () => void;
  onDismissFailed: (id: number) => void;
  onToggleFavorite: (record: GenerationRecord) => void;
}

/** Most recently completed jobs shown full-width in the queue itself, so a result is visible
 * right where it just finished without switching over to the result viewer. */
const MAX_COMPLETED_SHOWN = 5;

type DoneJob = Job & { record: GenerationRecord; imageUrl: string; kind: JobKind };

function isDone(job: Job): job is DoneJob {
  return job.status === 'done' && !!job.record && !!job.imageUrl;
}

function describe(job: Job): string {
  const { width, height, seed, steps, cfg, length } = job.params;
  const parts = [`${width}×${height}`];
  if (job.kind === 'video') parts.push(`${framesToSeconds(length ?? 81)}s`);
  parts.push(`seed ${seed}`);
  if (job.kind === 'image') parts.push(`${steps} steps`, `CFG ${cfg}`);
  return parts.join(' · ');
}

/** Right-hand, collapsible list of what is generating and what is waiting, with cancel controls. */
export default function QueuePanel({
  jobs,
  now,
  progressInfo,
  collapsed,
  onToggle,
  onCancelJob,
  onClearQueued,
  onDismissFailed,
  onToggleFavorite,
}: Props) {
  const running = jobs.find((j) => j.status === 'running');
  const queued = jobs.filter((j) => j.status === 'queued');
  const failed = jobs.filter((j) => j.status === 'failed' && !j.dismissed);
  const pending = queued.length + (running ? 1 : 0);

  // jobs is oldest-first (push order), so the most recently finished are at the end.
  const allDone = jobs.filter(isDone);
  const done = allDone.slice(-MAX_COMPLETED_SHOWN).reverse();
  const olderDoneCount = allDone.length - done.length;

  // How long everything still to run should take: what is left of the running job plus the
  // estimates of those waiting. Jobs without an estimate are left out and counted.
  let queueTotal: { text: string } | null = null;
  if (pending > 0) {
    let remaining = 0;
    let unknown = 0;
    if (running) {
      const display = describeProgress({
        progress: progressInfo?.progress ?? null,
        estimate: running.estimate ?? null,
        elapsedMs: now - (running.startedAt ?? now),
        lastStepAtMs: progressInfo?.lastStepAt != null ? progressInfo.lastStepAt - (running.startedAt ?? now) : null,
      });
      if (display.remainingMs !== null) remaining += display.remainingMs;
      else if (!display.overrunning) unknown++;
    }
    for (const job of queued) {
      if (job.estimate) remaining += job.estimate.totalMs;
      else unknown++;
    }
    if (remaining > 0 || unknown < pending) {
      queueTotal = {
        text: `about ${formatDuration(remaining)}${unknown > 0 ? ` (${unknown} without an estimate yet)` : ''}`,
      };
    }
  }

  if (collapsed) {
    return (
      <aside className="queue-panel queue-panel--collapsed">
        <button type="button" onClick={onToggle} title="Show the queue">
          ◀
          <span className="queue-panel__vertical">Queue</span>
          {pending > 0 && <span className="queue-panel__badge">{pending}</span>}
        </button>
      </aside>
    );
  }

  function batchLabel(job: Job): string | null {
    const batch = jobs.filter((j) => j.batchId === job.batchId);
    return batch.length > 1 ? `${batch.indexOf(job) + 1} of ${batch.length} in this batch` : null;
  }

  function renderJob(job: Job, extra: React.ReactNode) {
    const label = batchLabel(job);
    return (
      <div key={job.id} className={`queue-job queue-job--${job.status}`}>
        <div className="queue-job__top">
          <span className="queue-job__kind">{job.kind === 'video' ? '🎬' : '🖼️'}</span>
          <span className="queue-job__prompt" title={job.params.prompt}>
            {job.params.prompt}
          </span>
          {extra}
        </div>
        <div className="queue-job__meta">{describe(job)}</div>
        {label && <div className="queue-job__meta">{label}</div>}
        {job.status === 'queued' && job.estimate && (
          <div className="queue-job__meta">
            about {formatDuration(job.estimate.totalMs)}
            {job.estimate.loadMs ? ` (includes about ${formatDuration(job.estimate.loadMs)} loading models)` : ''}
          </div>
        )}
        {job.status === 'failed' && job.error && <div className="queue-job__error">{job.error}</div>}
      </div>
    );
  }

  return (
    <aside className="queue-panel">
      <div className="queue-panel__header">
        <strong>
          Queue
          {pending > 0 && <span className="queue-panel__badge">{pending}</span>}
        </strong>
        <button type="button" onClick={onToggle} title="Hide the queue">
          ▶
        </button>
      </div>

      <div className="queue-panel__body">
        {!running && queued.length === 0 && failed.length === 0 && done.length === 0 && (
          <p className="queue-panel__empty">
            Nothing queued. While something is generating, use "Queue Another" - or set a batch size to queue several
            at once.
          </p>
        )}

        {done.length > 0 && (
          <section>
            <div className="queue-panel__section-title">Recently completed</div>
            {done.map((job) => (
              <div key={job.id} className="queue-done">
                <div className="queue-done__media">
                  {job.kind === 'video' ? (
                    <GeneratedVideo src={job.imageUrl} filePath={job.record.imagePath} thumbnail />
                  ) : (
                    <img src={job.imageUrl} alt={job.record.prompt} />
                  )}
                  <button
                    type="button"
                    className={`library-card__fav${job.record.favorite ? ' library-card__fav--on' : ''}`}
                    onClick={() => onToggleFavorite(job.record)}
                    title={job.record.favorite ? 'Remove from favorites' : 'Add to favorites'}
                  >
                    {job.record.favorite ? '★' : '☆'}
                  </button>
                </div>
                <div className="queue-done__caption" title={job.params.prompt}>
                  {job.params.prompt}
                </div>
              </div>
            ))}
            {olderDoneCount > 0 && (
              <div className="queue-panel__hint">
                +{olderDoneCount} more this session - see Library &gt; Output
              </div>
            )}
          </section>
        )}

        {running && (
          <section>
            <div className="queue-panel__section-title">Generating now</div>
            {renderJob(
              running,
              <button type="button" className="queue-job__cancel" onClick={() => onCancelJob(running.id)} title="Cancel">
                ✕
              </button>
            )}
            <RunProgress compact job={running} progressInfo={progressInfo} now={now} />
          </section>
        )}

        {queued.length > 0 && (
          <section>
            <div className="queue-panel__section-title">
              Up next ({queued.length})
              <button type="button" className="queue-panel__clear" onClick={onClearQueued}>
                Clear all
              </button>
            </div>
            {queued.map((job, i) =>
              renderJob(
                job,
                <>
                  <span className="queue-job__position">#{i + 1}</span>
                  <button type="button" className="queue-job__cancel" onClick={() => onCancelJob(job.id)} title="Remove from queue">
                    ✕
                  </button>
                </>
              )
            )}
          </section>
        )}

        {queueTotal && (
          <div className="queue-panel__total">
            Queue total: {queueTotal.text}
          </div>
        )}

        {failed.length > 0 && (
          <section>
            <div className="queue-panel__section-title">Failed</div>
            {failed.map((job) =>
              renderJob(
                job,
                <button type="button" className="queue-job__cancel" onClick={() => onDismissFailed(job.id)} title="Dismiss">
                  ✕
                </button>
              )
            )}
          </section>
        )}
      </div>
    </aside>
  );
}
