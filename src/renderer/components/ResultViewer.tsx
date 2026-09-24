import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { GenerationRecord } from '../../shared/types';
import { Job, JobKind, ProgressInfo } from '../hooks/useGenerationQueue';
import { fitGrid } from '../utils/fitGrid';
import { timingSentence } from '../utils/timingText';
import RunProgress from './RunProgress';
import GeneratedVideo from './GeneratedVideo';
import ExpandButton from './Lightbox';

interface Props {
  /** The jobs of the batch being viewed, in order. */
  slots: Job[];
  now: number;
  progressInfo: ProgressInfo | null;
  onDelete: (record: GenerationRecord) => void;
  onToggleFavorite: (record: GenerationRecord) => void;
  onConvertToVideo: (record: GenerationRecord) => void;
  onCancelJob: (id: number) => void;
}

const GRID_GAP = 12;
// Padding of .result-viewer__body, and room left under the picture for its action buttons.
const BODY_PADDING = 12;
const ACTIONS_HEIGHT = 78;

function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setSize({ width: el.clientWidth, height: el.clientHeight }));
    observer.observe(el);
    setSize({ width: el.clientWidth, height: el.clientHeight });
    return () => observer.disconnect();
  }, []);
  return [ref, size] as const;
}

/**
 * Shows what a Generate run produced. One result looks like it always has; a batch can be paged
 * through one at a time (◀ ▶) or laid out as a grid sized to fill the whole area. Results appear
 * as their jobs finish.
 */
export default function ResultViewer({
  slots,
  now,
  progressInfo,
  onDelete,
  onToggleFavorite,
  onConvertToVideo,
  onCancelJob,
}: Props) {
  const [mode, setMode] = useState<'single' | 'grid'>('single');
  const [index, setIndex] = useState(0);
  const [bodyRef, bodySize] = useElementSize<HTMLDivElement>();

  // The last completed result actually shown, kept around so it stays on screen (full size)
  // while the next one generates instead of being replaced by a bare loading screen.
  const lastGoodRef = useRef<{ record: GenerationRecord; imageUrl: string; kind: JobKind; aspect: number } | null>(
    null
  );

  const lastDone = slots.reduce((last, job, i) => (job.status === 'done' ? i : last), -1);
  const batchKey = slots.length > 0 ? slots[0].batchId : null;
  const followRef = useRef(true);
  const shownBatch = useRef<number | null>(null);

  // Follow the newest finished result, unless the user has paged back to look at an earlier one.
  useEffect(() => {
    if (shownBatch.current !== batchKey) {
      shownBatch.current = batchKey;
      followRef.current = true;
      setIndex(Math.max(0, lastDone));
      return;
    }
    if (followRef.current && lastDone >= 0) setIndex(lastDone);
  }, [batchKey, lastDone]);

  const current = slots[Math.min(index, slots.length - 1)];
  const currentIndex = current ? slots.indexOf(current) : 0;
  const multiple = slots.length > 1;

  useEffect(() => {
    if (current?.status === 'done' && current.record && current.imageUrl) {
      lastGoodRef.current = {
        record: current.record,
        imageUrl: current.imageUrl,
        kind: current.kind,
        aspect: current.params.width / Math.max(current.params.height, 1),
      };
    }
  }, [current]);

  /** Deleting the currently-shown result must also drop it as the "keep showing while the next
   * one generates" fallback, or a later run would briefly show the just-deleted (now-missing) file. */
  function handleDelete(record: GenerationRecord) {
    if (lastGoodRef.current?.record.id === record.id) lastGoodRef.current = null;
    onDelete(record);
  }

  function goTo(i: number) {
    const next = Math.max(0, Math.min(slots.length - 1, i));
    setIndex(next);
    followRef.current = next === lastDone;
  }

  const doneCount = slots.filter((j) => j.status === 'done').length;
  const aspect = current ? current.params.width / Math.max(current.params.height, 1) : 1;

  function renderMedia(job: Job, thumbnail: boolean) {
    const url = job.imageUrl ?? '';
    const isVideo = job.kind === 'video';
    return isVideo ? (
      <GeneratedVideo
        src={url}
        filePath={job.record?.imagePath ?? ''}
        thumbnail={thumbnail}
        style={thumbnail ? undefined : { width: '100%', height: '100%', borderRadius: 8 }}
      />
    ) : (
      <img src={url} alt="Generated" />
    );
  }

  /** The status strip shown over the bottom of a still-in-progress job's fallback image. */
  function renderStatusOverlay(job: Job) {
    if (job.status === 'running') {
      return (
        <div className="result-media__overlay">
          <RunProgress compact job={job} progressInfo={progressInfo} now={now} />
          <button type="button" onClick={() => onCancelJob(job.id)}>
            ✕ Cancel
          </button>
        </div>
      );
    }
    if (job.status === 'queued') {
      return (
        <div className="result-media__overlay">
          <span className="result-media__overlay-text">Waiting in the queue...</span>
          <button type="button" onClick={() => onCancelJob(job.id)}>
            ✕ Cancel
          </button>
        </div>
      );
    }
    if (job.status === 'failed') {
      return (
        <div className="result-media__overlay result-media__overlay--failed">
          <span className="result-media__overlay-text">⚠ This generation failed: {job.error}</span>
        </div>
      );
    }
    return null;
  }

  function renderPending(job: Job) {
    if (job.status === 'running') {
      return (
        <div className="generate-preview__loading">
          <RunProgress job={job} progressInfo={progressInfo} now={now} />
          <button type="button" onClick={() => onCancelJob(job.id)}>
            ✕ Cancel
          </button>
        </div>
      );
    }
    if (job.status === 'failed') {
      return <div className="result-failed">⚠ This generation failed: {job.error}</div>;
    }
    return <div className="generate-preview__placeholder">Waiting in the queue...</div>;
  }

  function renderSingle() {
    if (current.status === 'done' && current.record) {
      const record = current.record;
      // Size the picture to the largest box of its own aspect ratio that fits, so the expand
      // button sits on the real corner of the picture rather than on a letterboxed container.
      const availableWidth = Math.max(0, bodySize.width - BODY_PADDING * 2);
      const availableHeight = Math.max(0, bodySize.height - BODY_PADDING * 2 - ACTIONS_HEIGHT);
      const mediaWidth = Math.floor(Math.min(availableWidth, availableHeight * aspect));
      const mediaHeight = Math.floor(mediaWidth / aspect);
      return (
        <div className="generate-preview__result">
          <div className="result-media" style={{ width: mediaWidth, height: mediaHeight }}>
            <ExpandButton
              src={current.imageUrl ?? ''}
              kind={current.kind}
              filePath={record.imagePath}
              alt={record.prompt}
            />
            {renderMedia(current, false)}
          </div>
          <div className="button-row">
            <button
              type="button"
              onClick={() => onToggleFavorite(record)}
              title={record.favorite ? 'Remove from favorites' : 'Save to favorites'}
            >
              {record.favorite ? '★ Favorited' : '☆ Favorite'}
            </button>
            {current.kind === 'image' && (
              <button
                type="button"
                onClick={() => onConvertToVideo(record)}
                title="Set up video mode with this image as the source"
              >
                🎬 Convert to Video
              </button>
            )}
            <button type="button" onClick={() => handleDelete(record)} title="Delete this generation and its file">
              🗑️ Delete
            </button>
          </div>
          {record.timing && <div className="result-timing">{timingSentence(record.timing)}</div>}
        </div>
      );
    }

    // Not done yet: keep showing the last completed result at full size (no room reserved for the
    // action row below it, so the picture gets the whole area) with the new job's status
    // overlaid on its bottom edge, instead of blanking the screen while the next one generates.
    const fallback = lastGoodRef.current;
    if (!fallback) return renderPending(current);
    const availableWidth = Math.max(0, bodySize.width - BODY_PADDING * 2);
    const availableHeight = Math.max(0, bodySize.height - BODY_PADDING * 2);
    const mediaWidth = Math.floor(Math.min(availableWidth, availableHeight * fallback.aspect));
    const mediaHeight = Math.floor(mediaWidth / fallback.aspect);
    return (
      <div className="generate-preview__result generate-preview__result--pending">
        <div className="result-media" style={{ width: mediaWidth, height: mediaHeight }}>
          <ExpandButton
            src={fallback.imageUrl}
            kind={fallback.kind}
            filePath={fallback.record.imagePath}
            alt={fallback.record.prompt}
          />
          {fallback.kind === 'video' ? (
            <GeneratedVideo
              src={fallback.imageUrl}
              filePath={fallback.record.imagePath}
              style={{ width: '100%', height: '100%', borderRadius: 8 }}
            />
          ) : (
            <img src={fallback.imageUrl} alt="Generated" />
          )}
          {renderStatusOverlay(current)}
        </div>
      </div>
    );
  }

  function renderGrid() {
    const { cols, tileWidth, tileHeight } = fitGrid(
      slots.length,
      aspect,
      bodySize.width - BODY_PADDING * 2,
      bodySize.height - BODY_PADDING * 2,
      GRID_GAP
    );
    return (
      <div
        className="result-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, ${tileWidth}px)`, gridAutoRows: `${tileHeight}px`, gap: GRID_GAP }}
      >
        {slots.map((job, i) => {
          const record = job.record;
          const done = job.status === 'done' && record;
          return (
            <div
              key={job.id}
              className={`result-tile${done ? ' result-tile--done' : ''}`}
              onClick={() => {
                if (!done) return;
                goTo(i);
                setMode('single');
              }}
            >
              {done ? (
                <>
                  {renderMedia(job, true)}
                  <ExpandButton src={job.imageUrl ?? ''} kind={job.kind} filePath={record.imagePath} alt={record.prompt} />
                  <button
                    type="button"
                    className={`library-card__fav${record.favorite ? ' library-card__fav--on' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleFavorite(record);
                    }}
                    title={record.favorite ? 'Remove from favorites' : 'Save to favorites'}
                  >
                    {record.favorite ? '★' : '☆'}
                  </button>
                </>
              ) : job.status === 'running' ? (
                <div className="result-tile__status">
                  <RunProgress compact job={job} progressInfo={progressInfo} now={now} />
                </div>
              ) : job.status === 'failed' ? (
                <div className="result-tile__status result-tile__status--failed" title={job.error}>
                  ⚠ Failed
                </div>
              ) : (
                <div className="result-tile__status">Queued</div>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="result-viewer">
      {multiple && (
        <div className="result-viewer__bar">
          <div className="result-viewer__pager">
            {mode === 'single' && (
              <>
                <button type="button" onClick={() => goTo(currentIndex - 1)} disabled={currentIndex === 0} title="Previous">
                  ◀
                </button>
                <span>
                  {currentIndex + 1} / {slots.length}
                </span>
                <button
                  type="button"
                  onClick={() => goTo(currentIndex + 1)}
                  disabled={currentIndex === slots.length - 1}
                  title="Next"
                >
                  ▶
                </button>
              </>
            )}
            <span className="result-viewer__progress">
              {doneCount} of {slots.length} done
            </span>
          </div>
          <div className="result-viewer__modes">
            <button type="button" className={mode === 'single' ? 'primary' : undefined} onClick={() => setMode('single')}>
              ▣ One at a time
            </button>
            <button type="button" className={mode === 'grid' ? 'primary' : undefined} onClick={() => setMode('grid')}>
              ▦ Grid
            </button>
          </div>
        </div>
      )}
      <div className="result-viewer__body" ref={bodyRef}>
        {/* Always rendered, even with nothing to show, so the size observer attaches on mount. */}
        {!current ? (
          <div className="generate-preview__placeholder">No image yet</div>
        ) : multiple && mode === 'grid' ? (
          renderGrid()
        ) : (
          renderSingle()
        )}
      </div>
    </div>
  );
}
