import { GenerationRecord } from '../../shared/types';
import { isUpscaleFamily } from '../../shared/upscale';
import { videoQualityFromCfg } from '../../shared/videoQuality';
import { Job, JobKind, ProgressInfo } from '../hooks/useGenerationQueue';
import { formatDuration } from '../utils/format';
import { describeProgress } from '../utils/progressModel';
import { framesToSeconds } from '../utils/video';
import ExpandButton from './Lightbox';
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
  /** Pull a completed result's exact prompt and settings back into the form. */
  onRerack: (record: GenerationRecord) => void;
}

/** Most recently completed jobs shown full-width in the queue itself, so a result is visible
 * right where it just finished without switching over to the result viewer. */
const MAX_COMPLETED_SHOWN = 5;

/** Job tiles on the folded bar before the rest are summed up as "+N". */
const MAX_BAR_TILES = 14;

type DoneJob = Job & { record: GenerationRecord; imageUrl: string; kind: JobKind };

function isDone(job: Job): job is DoneJob {
  return job.status === 'done' && !!job.record && !!job.imageUrl;
}

function describe(job: Job): string {
  const { width, height, seed, steps, cfg, length } = job.params;
  if (isUpscaleFamily(job.family)) return `Upscale to ${width}×${height} · ${job.params.upscaleModel ?? ''}`;
  const parts = [`${width}×${height}`];
  if (job.kind === 'video') parts.push(`${framesToSeconds(length ?? 81)}s`, `${videoQualityFromCfg(cfg)} quality`);
  parts.push(`seed ${seed}`);
  if (job.kind === 'image') parts.push(`${steps} steps`, `CFG ${cfg}`);
  return parts.join(' · ');
}

/** The queue bar along the bottom of the window: a slim status strip while folded, and a drawer of
 * horizontal cards (generating, waiting, finished, failed) with cancel controls while open. */
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
  onRerack,
}: Props) {
  const running = jobs.find((j) => j.status === 'running');
  const queued = jobs.filter((j) => j.status === 'queued');
  const failed = jobs.filter((j) => j.status === 'failed' && !j.dismissed);
  const pending = queued.length + (running ? 1 : 0);

  // jobs is oldest-first (push order), so the most recently finished are at the end.
  const allDone = jobs.filter(isDone);
  const done = allDone.slice(-MAX_COMPLETED_SHOWN).reverse();
  const olderDoneCount = allDone.length - done.length;

  const runningDisplay = running
    ? describeProgress({
        progress: progressInfo?.progress ?? null,
        estimate: running.estimate ?? null,
        elapsedMs: now - (running.startedAt ?? now),
        lastStepAtMs: progressInfo?.lastStepAt != null ? progressInfo.lastStepAt - (running.startedAt ?? now) : null,
      })
    : null;

  // How long everything still to run should take: what is left of the running job plus the
  // estimates of those waiting. Jobs without an estimate are left out and counted.
  let queueTotal: { text: string } | null = null;
  if (pending > 0) {
    let remaining = 0;
    let unknown = 0;
    if (runningDisplay) {
      if (runningDisplay.remainingMs !== null) remaining += runningDisplay.remainingMs;
      else if (!runningDisplay.overrunning) unknown++;
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

  // The one line the folded bar shows: what the running job is doing, or how things stand.
  let statusText = pending > 0 ? '' : failed.length > 0 ? `${failed.length} failed` : 'Nothing queued';
  if (runningDisplay) {
    statusText = runningDisplay.label;
    if (runningDisplay.overrunning) statusText += ' · taking longer than usual';
    else if (runningDisplay.remainingMs !== null) statusText += ` · about ${formatDuration(runningDisplay.remainingMs)} left`;
  }

  function batchLabel(job: Job): string | null {
    const batch = jobs.filter((j) => j.batchId === job.batchId);
    return batch.length > 1 ? `${batch.indexOf(job) + 1} of ${batch.length} in this batch` : null;
  }

  function renderJob(job: Job, extra: React.ReactNode, below?: React.ReactNode) {
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
        {below}
      </div>
    );
  }

  const tiles = [...(running ? [running] : []), ...queued, ...failed];
  const shownTiles = tiles.slice(0, MAX_BAR_TILES);
  const hiddenTiles = tiles.length - shownTiles.length;

  return (
    <section className={`queue-bar${collapsed ? ' queue-bar--collapsed' : ''}`}>
      {collapsed && runningDisplay && (
        <div
          className={`queue-bar__line${runningDisplay.determinate ? '' : ' queue-bar__line--pulse'}`}
          style={{ '--queue-bar-fill': `${Math.round((runningDisplay.determinate ? runningDisplay.fraction : 1) * 100)}%` } as React.CSSProperties}
        />
      )}
      {/* Folded, the whole bar opens the queue; its button is there for the keyboard (the click
          bubbles up). Open, only the ✕ closes it. */}
      <div className="queue-bar__head" onClick={collapsed ? onToggle : undefined}>
        <button
          type="button"
          className="queue-bar__toggle"
          onClick={collapsed ? undefined : onToggle}
          title={collapsed ? 'Show the queue' : 'Hide the queue'}
          aria-expanded={!collapsed}
        >
          {collapsed ? '▲' : '✕'}
        </button>
        <strong className="queue-bar__title">
          Queue
          {pending > 0 && <span className="queue-panel__badge">{pending}</span>}
        </strong>
        {collapsed ? (
          <>
            <div className="queue-bar__tiles">
              {shownTiles.map((job) => {
                const isRunning = job.status === 'running';
                const position = job.status === 'queued' ? queued.indexOf(job) + 1 : 0;
                const status = isRunning
                  ? (runningDisplay?.label ?? 'Generating')
                  : job.status === 'failed'
                    ? 'Failed'
                    : `Waiting - #${position}`;
                const fillStyle =
                  isRunning && runningDisplay?.determinate
                    ? ({ '--queue-tile-fill': `${Math.round(runningDisplay.fraction * 100)}%` } as React.CSSProperties)
                    : undefined;
                return (
                  <span
                    key={job.id}
                    className={`queue-bar__tile queue-bar__tile--${job.status}${
                      isRunning && !runningDisplay?.determinate ? ' queue-bar__tile--pulse' : ''
                    }`}
                    style={fillStyle}
                    title={`${status}\n${job.params.prompt}`}
                  >
                    <span className="queue-bar__tile-icon">{job.status === 'failed' ? '⚠' : job.kind === 'video' ? '🎬' : '🖼️'}</span>
                    {position > 0 && <span className="queue-bar__tile-position">{position}</span>}
                  </span>
                );
              })}
              {hiddenTiles > 0 && <span className="queue-bar__more">+{hiddenTiles}</span>}
            </div>
            <span className="queue-bar__status">{statusText}</span>
          </>
        ) : (
          <>
            {queueTotal && <span className="queue-bar__status">Queue total: {queueTotal.text}</span>}
            {queued.length > 0 && (
              <button type="button" className="queue-panel__clear queue-bar__clear" onClick={onClearQueued}>
                Clear all
              </button>
            )}
          </>
        )}
      </div>

      {!collapsed && (
        <div className="queue-bar__body">
          {!running && queued.length === 0 && failed.length === 0 && done.length === 0 && (
            <p className="queue-panel__empty">
              Nothing queued. While something is generating, use "Queue Another" - or set a batch size to queue several at
              once.
            </p>
          )}

          {running && (
            <section className="queue-bar__group">
              <div className="queue-panel__section-title">Generating now</div>
              <div className="queue-bar__cards">
                {renderJob(
                  running,
                  <button type="button" className="queue-job__cancel" onClick={() => onCancelJob(running.id)} title="Cancel">
                    ✕
                  </button>,
                  <RunProgress compact job={running} progressInfo={progressInfo} now={now} />
                )}
              </div>
            </section>
          )}

          {queued.length > 0 && (
            <section className="queue-bar__group">
              <div className="queue-panel__section-title">Up next ({queued.length})</div>
              <div className="queue-bar__cards">
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
              </div>
            </section>
          )}

          {done.length > 0 && (
            <section className="queue-bar__group">
              <div className="queue-panel__section-title">
                Recently completed
                {olderDoneCount > 0 && (
                  <span className="queue-panel__hint" title="Older ones are in Library > Output">
                    +{olderDoneCount} more
                  </span>
                )}
              </div>
              <div className="queue-bar__cards">
                {done.map((job) => (
                  <div key={job.id} className="queue-done">
                    <div className="queue-done__media">
                      {job.kind === 'video' ? (
                        <GeneratedVideo src={job.imageUrl} filePath={job.record.imagePath} thumbnail />
                      ) : (
                        <img src={job.imageUrl} alt={job.record.prompt} />
                      )}
                      <ExpandButton src={job.imageUrl} kind={job.kind} filePath={job.record.imagePath} alt={job.record.prompt} />
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
                    {!isUpscaleFamily(job.family) && (
                      <button
                        type="button"
                        className="queue-done__rerack"
                        onClick={() => onRerack(job.record)}
                        title="Load this prompt and its exact settings back into the form"
                      >
                        ↺ Re-rack
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {failed.length > 0 && (
            <section className="queue-bar__group">
              <div className="queue-panel__section-title">Failed</div>
              <div className="queue-bar__cards">
                {failed.map((job) =>
                  renderJob(
                    job,
                    <button type="button" className="queue-job__cancel" onClick={() => onDismissFailed(job.id)} title="Dismiss">
                      ✕
                    </button>
                  )
                )}
              </div>
            </section>
          )}
        </div>
      )}
    </section>
  );
}
