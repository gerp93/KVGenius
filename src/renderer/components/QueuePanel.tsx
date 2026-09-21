import { Job } from '../hooks/useGenerationQueue';
import { formatElapsed } from '../utils/format';
import { framesToSeconds } from '../utils/video';

interface Props {
  jobs: Job[];
  now: number;
  collapsed: boolean;
  onToggle: () => void;
  onCancelJob: (id: number) => void;
  onClearQueued: () => void;
  onDismissFailed: (id: number) => void;
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
export default function QueuePanel({ jobs, now, collapsed, onToggle, onCancelJob, onClearQueued, onDismissFailed }: Props) {
  const running = jobs.find((j) => j.status === 'running');
  const queued = jobs.filter((j) => j.status === 'queued');
  const failed = jobs.filter((j) => j.status === 'failed' && !j.dismissed);
  const pending = queued.length + (running ? 1 : 0);

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
        {!running && queued.length === 0 && failed.length === 0 && (
          <p className="queue-panel__empty">
            Nothing queued. While something is generating, use "Queue Another" - or set a batch size to queue several
            at once.
          </p>
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
            <div className="progress-bar progress-bar--indeterminate" />
            <div className="queue-job__meta">{formatElapsed(Math.floor((now - (running.startedAt ?? now)) / 1000))} elapsed</div>
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
