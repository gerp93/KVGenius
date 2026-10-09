import { useLayoutEffect, useRef, useState } from 'react';
import { GenerationRecord } from '../../shared/types';
import { sourceMissingMessage } from '../../shared/sourceFamilies';
import { UPSCALE_FAMILY, isUpscaleFamily } from '../../shared/upscale';
import { keptFilesOf, useMissingSources } from '../hooks/useMissingSources';
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
  /** The finished result whose details panel is open, if any (its card is highlighted). */
  activeDetailsId: number | null;
  /** Open (or close) the details panel for a finished result. */
  onOpenDetails: (record: GenerationRecord) => void;
  /** A short confirmation of the last action taken on a finished result (pinned, moved to the Trash). */
  notice: string | null;
  onToggleFavorite: (record: GenerationRecord) => void;
  onTogglePinned: (record: GenerationRecord) => void;
  /** Move a finished result to the Trash. */
  onDelete: (record: GenerationRecord) => void;
  /** Pull a completed result's exact prompt and settings back into the form. */
  onRerack: (record: GenerationRecord) => void;
}

/** Most recently completed jobs shown full-width in the queue itself, so a result is visible
 * right where it just finished without switching over to the result viewer. */
const MAX_COMPLETED_SHOWN = 20;

/** Size of a job tile on the folded bar and the gap between them (keep in step with `.queue-bar__tile`
 * and `.queue-bar__tiles` in index.css). */
const TILE_SIZE = 28;
const TILE_GAP = 6;

type DoneJob = Job & { record: GenerationRecord; imageUrl: string; kind: JobKind };

function isDone(job: Job): job is DoneJob {
  // A recalled record (Re-rack) was never run, so it is not a completed job.
  return job.status === 'done' && !job.isRecall && !!job.record && !!job.imageUrl;
}

function describe(job: Job): string {
  const { width, height, seed, steps, cfg, length } = job.params;
  if (isUpscaleFamily(job.family)) return `Upscale to ${width}×${height} · ${job.params.upscaleModel ?? ''}`;
  const parts = [`${width}×${height}`];
  if (job.kind === 'video') parts.push(`${framesToSeconds(length ?? 81)}s`, `${videoQualityFromCfg(cfg)} quality`);
  parts.push(`seed ${seed}`);
  if (job.kind === 'image') parts.push(`${steps} steps`, `CFG ${cfg}`);
  if (job.params.outpaint) parts.push('outpaint');
  else if (job.params.denoise !== undefined) parts.push(`${job.params.maskImagePath ? 'inpaint' : 'image to image'} ${job.params.denoise.toFixed(2)}`);
  return parts.join(' · ');
}

/** The picture (or, for a video upscale, the video's first frame) a queued job works from, small, in the bottom corner of its card.
 * It hides itself if the file can no longer be loaded. */
function ReferenceThumb({ path, video }: { path: string; video: boolean }) {
  const [broken, setBroken] = useState(false);
  if (broken) return null;
  const url = window.kvgenius.imageUrlFor(path);
  return video ? (
    <video
      className="queue-job__reference"
      src={`${url}#t=0.1`}
      muted
      preload="metadata"
      playsInline
      title="The video this is made from"
      onError={() => setBroken(true)}
    />
  ) : (
    <img
      className="queue-job__reference"
      src={url}
      alt="Reference image"
      title="The reference image this is made from"
      draggable={false}
      onError={() => setBroken(true)}
    />
  );
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
  notice,
  activeDetailsId,
  onOpenDetails,
  onToggleFavorite,
  onTogglePinned,
  onDelete,
  onRerack,
}: Props) {
  // How many tiles the folded bar has room for, kept up to date as the window or status text changes.
  const tilesRef = useRef<HTMLDivElement>(null);
  const [tileSlots, setTileSlots] = useState(12);
  useLayoutEffect(() => {
    const el = tilesRef.current;
    if (!el) return;
    const measure = () => setTileSlots(Math.max(1, Math.floor((el.clientWidth + TILE_GAP) / (TILE_SIZE + TILE_GAP))));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [collapsed]);

  const running = jobs.find((j) => j.status === 'running');
  const queued = jobs.filter((j) => j.status === 'queued');
  const failed = jobs.filter((j) => j.status === 'failed' && !j.dismissed);
  const pending = queued.length + (running ? 1 : 0);

  // jobs is oldest-first (push order), so the most recently finished are at the end.
  const allDone = jobs.filter(isDone);
  const done = allDone.slice(-MAX_COMPLETED_SHOWN).reverse();
  // Results made from a picture can only be re-racked while their kept copy of it still exists.
  const missingSources = useMissingSources(done.flatMap((job) => keptFilesOf(job.record)));
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
    // Anything made from a picture (a video, image to image, inpainting, outpainting, a picture upscale) shows it in the corner;
    // a video upscale shows the first frame of the video it enlarges.
    const reference = job.params.sourceImagePath ?? job.params.sourceVideoPath;
    const referenceIsVideo = !job.params.sourceImagePath && !!job.params.sourceVideoPath;
    return (
      <div key={job.id} className={`queue-job queue-job--${job.status}${reference ? ' queue-job--has-reference' : ''}`}>
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
        {reference && <ReferenceThumb path={reference} video={referenceIsVideo} />}
      </div>
    );
  }

  const tiles = [...(running ? [running] : []), ...queued, ...failed];
  // As many tiles as fit across the folded bar; when they do not all fit, the last slot is the "+N".
  const fits = tiles.length <= tileSlots;
  const shownTiles = fits ? tiles : tiles.slice(0, Math.max(0, tileSlots - 1));
  const hiddenTiles = tiles.length - shownTiles.length;

  return (
    <section className={`queue-bar${collapsed ? ' queue-bar--collapsed' : ''}`}>
      {collapsed && runningDisplay && (
        <div
          className={`queue-bar__line${runningDisplay.determinate ? '' : ' queue-bar__line--pulse'}`}
          style={{ '--queue-bar-fill': `${Math.round((runningDisplay.determinate ? runningDisplay.fraction : 1) * 100)}%` } as React.CSSProperties}
        />
      )}
      {/* The whole bar opens and closes the queue; its button is there for the keyboard (the click
          bubbles up). Buttons of their own inside the bar (Clear all) stop the click so they do not also fold it. */}
      <div className="queue-bar__head" onClick={onToggle}>
        <button
          type="button"
          className="queue-bar__toggle"
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
            <div className="queue-bar__tiles" ref={tilesRef}>
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
            <span className="queue-bar__status">{notice ?? statusText}</span>
          </>
        ) : (
          <>
            {(notice || queueTotal) && (
              <span className="queue-bar__status">{notice ?? `Queue total: ${queueTotal?.text}`}</span>
            )}
            {queued.length > 0 && (
              <button
                type="button"
                className="queue-panel__clear queue-bar__clear"
                onClick={(e) => {
                  e.stopPropagation();
                  onClearQueued();
                }}
              >
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
            <section className="queue-bar__group queue-bar__group--done">
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
                  <div key={job.id} className={`queue-done${activeDetailsId === job.record.id ? ' queue-done--active' : ''}`}>
                    <div
                      className="queue-done__media queue-done__open"
                      title="Open the details"
                      onClick={(e) => {
                        // The expand and favorite buttons on the picture do their own thing.
                        if (!(e.target as HTMLElement).closest('button')) onOpenDetails(job.record);
                      }}
                    >
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
                    <div className="queue-done__caption queue-done__open" title={job.params.prompt} onClick={() => onOpenDetails(job.record)}>
                      {job.params.prompt}
                    </div>
                    <div className="queue-done__actions">
                      {(!isUpscaleFamily(job.family) || job.family === UPSCALE_FAMILY) && (
                        <button
                          type="button"
                          className="queue-done__rerack"
                          onClick={() => onRerack(job.record)}
                          disabled={keptFilesOf(job.record).some((file) => missingSources.has(file))}
                          title={
                            keptFilesOf(job.record).some((file) => missingSources.has(file))
                              ? sourceMissingMessage(job.record?.modelFamily ?? '')
                              : job.family === UPSCALE_FAMILY
                                ? 'Open the original in Tools > Upscale'
                                : 'Load this prompt and its exact settings back into the form'
                          }
                        >
                          ↺ Re-rack
                        </button>
                      )}
                      <button
                        type="button"
                        className={`queue-done__icon${job.record.pinned ? ' queue-done__icon--on' : ''}`}
                        onClick={() => onTogglePinned(job.record)}
                        title={
                          job.record.pinned
                            ? 'Unpin - remove this from Library > Prompts'
                            : 'Pin as the example of this prompt, shown under Library > Prompts'
                        }
                      >
                        📌
                      </button>
                      <button
                        type="button"
                        className="queue-done__icon"
                        onClick={() => onDelete(job.record)}
                        title="Delete - moves it to the Trash (restore it from Library > Trash)"
                      >
                        🗑️
                      </button>
                    </div>
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
