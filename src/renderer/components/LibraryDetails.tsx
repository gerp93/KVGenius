import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DEFAULT_GIF_FPS, DEFAULT_GIF_WIDTH, GIF_FPS_CHOICES, GIF_WIDTHS } from '../../shared/gif';
import { UPSCALE_FACTORS, DEFAULT_UPSCALE_FACTOR, UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY } from '../../shared/upscale';
import { FAMILY_KIND, GenerationKind, GenerationRecord, VideoSourceRequest } from '../../shared/types';
import { isExtendableFamily } from '../../shared/textToVideo';
import { GenerationQueue, MAX_PENDING_JOBS } from '../hooks/useGenerationQueue';
import { formatBytes, formatDifference, formatDuration } from '../utils/format';
import { shouldSplitDetails } from '../../shared/detailsLayout';
import { SOURCE_MISSING_MESSAGE } from '../../shared/sourceFamilies';
import { useSourceMissing } from '../hooks/useMissingSources';
import { useRetryWhenReachable } from '../hooks/useRetryWhenReachable';
import { isUpscale } from '../utils/library';
import CopyButton from './CopyButton';
import GeneratedVideo from './GeneratedVideo';
import OriginBadge from './OriginBadge';
import ExpandButton from './Lightbox';

// wan22-i2v's frame rate (see Generate.tsx) - only used to show a video's length in seconds.
const VIDEO_FPS = 16;

// The upscale choices are kept for the session, so closing the panel and opening another item does
// not send you back to the first model.
let lastUpscaleModel = '';
let lastUpscaleFactor = DEFAULT_UPSCALE_FACTOR;

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

/** GIFs made from a video are stored as images, but can't be upscaled or animated again. */
function isGif(record: GenerationRecord): boolean {
  return record.imagePath.toLowerCase().endsWith('.gif');
}

interface Props {
  record: GenerationRecord;
  queue: GenerationQueue;
  onClose: () => void;
  /** Open the picture in the full-window viewer. */
  onExpand: () => void;
  onToggleFavorite: (record: GenerationRecord) => void;
  onTogglePinned: (record: GenerationRecord) => void;
  onToggleHidden: (record: GenerationRecord) => void;
  onDelete: (record: GenerationRecord) => void;
  /** Load the generation into a Generate tab. */
  onRerack: (record: GenerationRecord) => void;
  onImageToVideo: (record: GenerationRecord) => void;
  /** Start an image to image run from this picture (opens Generate with it as the source image). */
  onImageToImage: (record: GenerationRecord) => void;
  /** Continue a video from its last frame (opens Generate with that frame as the source image; the new clip will be joined onto the video). */
  onExtendVideo: (request: VideoSourceRequest) => void;
  onSaveAs: (record: GenerationRecord) => void;
  onReveal: (record: GenerationRecord) => void;
  /** An upscale was added to the queue (the page can show its queue). */
  onUpscaleQueued: () => void;
  /** A GIF was made from this video: the new image's record, for the page to fold into its list. */
  onGifMade: (made: GenerationRecord) => void;
  onError: (message: string | null) => void;
  onNotice: (message: string | null) => void;
}

/**
 * The Library's details side panel for one generation: its picture, the actions on it (re-rack,
 * favorite, pin, hide, GIF, upscale, ...), the prompt with a copy button, and everything known about
 * it. Shared by Library > Output and Library > Prompts so they behave alike; the page owns the list
 * and decides what each action does to it.
 */
export default function LibraryDetails({
  record,
  queue,
  onClose,
  onExpand,
  onToggleFavorite,
  onTogglePinned,
  onToggleHidden,
  onDelete,
  onRerack,
  onImageToVideo,
  onImageToImage,
  onExtendVideo,
  onSaveAs,
  onReveal,
  onUpscaleQueued,
  onGifMade,
  onError,
  onNotice,
}: Props) {
  const [size, setSize] = useState<number | null>(null);
  // Extending a video: its last frame is read first (it needs ffmpeg), then Generate opens with it.
  const [extending, setExtending] = useState(false);
  async function handleExtend() {
    setExtending(true);
    onError(null);
    try {
      const frame = await window.kvgenius.prepareVideoExtension(record.id);
      onExtendVideo({ imagePath: frame.path, width: frame.width, height: frame.height, extend: { fromId: record.id, prompt: record.prompt } });
    } catch (err) {
      onError(err instanceof Error ? err.message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '') : String(err));
    } finally {
      setExtending(false);
    }
  }
  // A video or upscale made from a picture keeps a copy of it; if that copy is gone it cannot be re-run.
  const sourceMissing = useSourceMissing(record);
  // Upscale controls: the models come from ComfyUI when the panel opens.
  const [upscaleModels, setUpscaleModels] = useState<string[] | null>(null);
  const [upscaleModel, setUpscaleModelState] = useState(lastUpscaleModel);
  const [upscaleFactor, setUpscaleFactorState] = useState(lastUpscaleFactor);
  // ComfyUI did not answer when the models were asked for.
  const [upscaleUnreachable, setUpscaleUnreachable] = useState(false);
  // GIF conversion controls, for a video.
  const [gifWidth, setGifWidth] = useState(DEFAULT_GIF_WIDTH);
  const [gifFps, setGifFps] = useState(DEFAULT_GIF_FPS);
  const [makingGif, setMakingGif] = useState(false);
  // Two-column layout once the panel is big enough for it to pay off (see shared/detailsLayout.ts): the
  // picture gets a full-height column of its own beside the details. It depends on the panel's actual
  // size and the picture's shape, so the panel is measured.
  const panelRef = useRef<HTMLElement>(null);
  const [panelSize, setPanelSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const measure = () => setPanelSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const split = shouldSplitDetails({
    panelWidth: panelSize.width,
    panelHeight: panelSize.height,
    aspect: record.width / Math.max(record.height, 1),
  });

  function setUpscaleModel(model: string) {
    lastUpscaleModel = model;
    setUpscaleModelState(model);
  }

  function setUpscaleFactor(factor: number) {
    lastUpscaleFactor = factor;
    setUpscaleFactorState(factor);
  }

  const imagePath = record.imagePath;
  useEffect(() => {
    setSize(null);
    let cancelled = false;
    window.kvgenius
      .getFileSize(imagePath)
      .then((bytes) => {
        if (!cancelled) setSize(bytes);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [imagePath]);

  /** Asks ComfyUI which upscale models it has. If it cannot be reached, says so both at the top of the page
   * (onError) and right under the Upscale controls, where the person is looking. */
  function loadUpscaleModels(isCancelled: () => boolean = () => false) {
    setUpscaleUnreachable(false);
    window.kvgenius
      .listUpscaleModels()
      .then((models) => {
        if (isCancelled()) return;
        setUpscaleModels(models);
        setUpscaleModelState((prev) => {
          const next = models.includes(prev) ? prev : (models[0] ?? '');
          lastUpscaleModel = next;
          return next;
        });
      })
      .catch(() => {
        if (isCancelled()) return;
        setUpscaleUnreachable(true);
        onError('Could not reach ComfyUI to list upscale models.');
      });
  }

  useEffect(() => {
    let cancelled = false;
    loadUpscaleModels(() => cancelled);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ComfyUI was not running when the panel opened: once it is started, the models appear by themselves (and the notice at the top goes).
  useRetryWhenReachable(upscaleUnreachable, () => {
    loadUpscaleModels();
    onError(null);
  });

  /** Adds an upscale job to the shared queue; several can be waiting at once. */
  function handleUpscale() {
    if (!upscaleModel) return;
    onNotice(null);
    onError(null);
    onUpscaleQueued();
    const isVideo = kindOf(record) === 'video';
    // Video encoders need even dimensions.
    const scaled = (n: number) => (isVideo ? 2 * Math.round((n * upscaleFactor) / 2) : Math.round(n * upscaleFactor));
    const added = queue.enqueue([
      {
        family: isVideo ? UPSCALE_VIDEO_FAMILY : UPSCALE_FAMILY,
        kind: isVideo ? 'video' : 'image',
        params: {
          prompt: record.prompt,
          width: scaled(record.width),
          height: scaled(record.height),
          seed: record.seed,
          steps: record.steps,
          cfg: record.cfg,
          ...(isVideo ? { length: record.length ?? undefined, sourceVideoPath: record.imagePath } : { sourceImagePath: record.imagePath }),
          upscaleModel,
        },
      },
    ]);
    if (added === 0) onError(`The queue is full (${MAX_PENDING_JOBS} waiting).`);
  }

  async function handleMakeGif() {
    setMakingGif(true);
    onNotice(null);
    onError(null);
    try {
      const { record: made } = await window.kvgenius.convertToGif(record.id, { fps: gifFps, width: gifWidth });
      onNotice(`Made a ${made.width} × ${made.height} GIF - saved as a new image.`);
      onGifMade(made);
    } catch (err) {
      // Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method".
      const message = err instanceof Error ? err.message : String(err);
      onError(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''));
    } finally {
      setMakingGif(false);
    }
  }

  const header = (
    <div className="library-panel__header">
      <span className="library-panel__title">
        <strong>Details</strong>
        <OriginBadge record={record} />
      </span>
      <span className="library-panel__window-buttons">
        <button type="button" onClick={onClose} title="Close">
          ✕
        </button>
      </span>
      <span className="library-panel__header-buttons">
        <button type="button" onClick={() => onToggleFavorite(record)}>
          {record.favorite ? '★ Favorited' : '☆ Favorite'}
        </button>
        <button
          type="button"
          onClick={() => onTogglePinned(record)}
          title={
            record.pinned
              ? 'Unpin - remove this from Library > Prompts'
              : 'Pin as the example of this prompt, shown under Library > Prompts'
          }
        >
          {record.pinned ? '📌 Pinned' : '📌 Pin'}
        </button>
        <button type="button" onClick={() => onToggleHidden(record)}>
          {record.hidden ? 'Unhide' : 'Hide'}
        </button>
      </span>
    </div>
  );
  const media = (
    <div className="library-panel__media">
      <button type="button" className="expand-button" title="Expand" onClick={onExpand}>
        ⤢
      </button>
      {kindOf(record) === 'video' ? (
        <GeneratedVideo src={window.kvgenius.imageUrlFor(record.imagePath)} filePath={record.imagePath} />
      ) : (
        <img src={window.kvgenius.imageUrlFor(record.imagePath)} alt={record.prompt} />
      )}
    </div>
  );
  const rest = (
    <>
      <button
        type="button"
        className="primary"
        onClick={() => onRerack(record)}
        disabled={sourceMissing}
        title={sourceMissing ? SOURCE_MISSING_MESSAGE : undefined}
        style={{ width: '100%' }}
      >
        ↺ Re-rack
      </button>
      {sourceMissing && (
        <p className="library-panel__warning" role="alert">
          {SOURCE_MISSING_MESSAGE}
        </p>
      )}
      {kindOf(record) === 'video' && isExtendableFamily(record.modelFamily) && (
        <>
          <button
            type="button"
            onClick={() => void handleExtend()}
            disabled={extending}
            title="Make more of this video: a new clip carries on from its last frame and is joined onto the end of it"
            style={{ width: '100%' }}
          >
            {extending ? 'Reading the last frame...' : '➕ Extend this video'}
          </button>
        </>
      )}
      {kindOf(record) === 'video' && (
        <div className="library-panel__upscale">
          <span className="field-label" style={{ margin: 0 }}>
            GIF
          </span>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <select
              value={gifWidth}
              onChange={(e) => setGifWidth(Number(e.target.value))}
              disabled={makingGif}
              style={{ flex: 1, minWidth: 0 }}
              title="Maximum width"
            >
              {GIF_WIDTHS.map((w) => (
                <option key={w} value={w}>
                  {w}px wide
                </option>
              ))}
            </select>
            <select value={gifFps} onChange={(e) => setGifFps(Number(e.target.value))} disabled={makingGif} title="Frames per second">
              {GIF_FPS_CHOICES.map((f) => (
                <option key={f} value={f}>
                  {f} fps
                </option>
              ))}
            </select>
            <button type="button" onClick={handleMakeGif} disabled={makingGif}>
              {makingGif ? 'Converting...' : 'Make GIF'}
            </button>
          </div>
        </div>
      )}
      {!isGif(record) && (
        <div className="library-panel__upscale">
          <span className="field-label" style={{ margin: 0 }}>
            Upscale
          </span>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <select
              value={upscaleModel}
              onChange={(e) => setUpscaleModel(e.target.value)}
              style={{ flex: 1, minWidth: 0 }}
              title="Upscale model"
            >
              {upscaleModels === null && <option value="">Choose model...</option>}
              {upscaleModels?.length === 0 && <option value="">No upscale models installed</option>}
              {upscaleModels?.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <select value={upscaleFactor} onChange={(e) => setUpscaleFactor(Number(e.target.value))} title="Size multiplier">
              {UPSCALE_FACTORS.map((f) => (
                <option key={f} value={f}>
                  {f}×
                </option>
              ))}
            </select>
            <button type="button" onClick={handleUpscale} disabled={!upscaleModel || upscaleUnreachable}>
              Upscale
            </button>
          </div>
          {upscaleUnreachable && (
            <p className="library-panel__warning" role="alert">
              ⚠ ComfyUI is not reachable, so there are no upscale models to choose from. Start ComfyUI (top right), then{' '}
              <button type="button" className="link-button" onClick={() => loadUpscaleModels()}>
                try again
              </button>
              .
            </p>
          )}
        </div>
      )}
      <div className="library-panel__actions">
        {kindOf(record) === 'image' && !isGif(record) && (
          <button type="button" onClick={() => onImageToVideo(record)} title="Make a video starting from this picture">
            🎬 Image to video
          </button>
        )}
        {kindOf(record) === 'image' && !isGif(record) && (
          <button type="button" onClick={() => onImageToImage(record)} title="Start a new picture from this one (image to image)">
            🎨 Image to image
          </button>
        )}
        {kindOf(record) === 'image' && (
          <CopyButton
            imagePath={record.imagePath}
            compact
            title={isGif(record) ? 'Copy the image (an animated GIF copies as one still frame)' : 'Copy the image'}
          />
        )}
        <button type="button" onClick={() => onSaveAs(record)} title="Save As...">
          💾
        </button>
        <button type="button" onClick={() => onReveal(record)} title="Show in File Manager">
          📂
        </button>
        <button type="button" onClick={() => onDelete(record)} title="Delete (moves to the Trash)">
          🗑️
        </button>
      </div>

      <div className="library-panel__prompt-header">
        <span className="field-label" style={{ margin: 0 }}>
          Prompt
        </span>
        <CopyButton compact className="copy-button--icon" text={record.prompt} title="Copy this prompt" />
      </div>
      <p className="library-panel__prompt">{record.prompt}</p>

      <dl className="library-panel__meta">
        <dt>Type</dt>
        <dd>{kindOf(record) === 'video' ? 'Video' : 'Image'}</dd>
        <dt>Model</dt>
        <dd>{isUpscale(record) ? `Upscale (${record.modelFamily}) - enlarged from another picture` : record.modelFamily}</dd>
        {record.modelName && record.modelSettings && (
          <>
            <dt>Variant</dt>
            <dd title={Object.values(record.modelSettings.files).join('\n')}>
              {record.modelName}
              {record.modelSettings.sampler && record.modelSettings.scheduler ? ` - ${record.modelSettings.sampler} / ${record.modelSettings.scheduler}` : ''}
            </dd>
          </>
        )}
        {record.styleName && (
          <>
            <dt>Style</dt>
            <dd title="The style's words are already part of the prompt above">{record.styleName}</dd>
          </>
        )}
        <dt>Dimensions</dt>
        <dd>
          {record.width} × {record.height}
        </dd>
        <dt>File size</dt>
        <dd>{size === null ? '-' : formatBytes(size)}</dd>
        {record.length !== null && (
          <>
            <dt>Length</dt>
            <dd>
              {Math.round(((record.length - 1 + (record.extendedFrames ?? 0)) / VIDEO_FPS) * 4) / 4}s ({record.length + (record.extendedFrames ?? 0)} frames)
              {record.extendedFrames !== null && ` - an earlier video of ${record.extendedFrames} frames, extended by ${record.length}`}
            </dd>
          </>
        )}
        <dt>Seed</dt>
        <dd>{record.seed}</dd>
        {kindOf(record) === 'image' && (
          <>
            <dt>Steps</dt>
            <dd>{record.steps}</dd>
            <dt>CFG</dt>
            <dd>{record.cfg}</dd>
          </>
        )}
        {record.timing && (
          <>
            <dt>Estimated</dt>
            <dd>{record.timing.estimateMs === null ? 'no estimate yet' : formatDuration(record.timing.estimateMs)}</dd>
            <dt>Took</dt>
            <dd>
              {formatDuration(record.timing.actualMs)}
              {record.timing.loadMs !== null && record.timing.loadMs >= 2000
                ? ` (${formatDuration(record.timing.loadMs)} loading models)`
                : ''}
            </dd>
            {record.timing.estimateMs !== null && (
              <>
                <dt>Difference</dt>
                <dd>{formatDifference(record.timing.estimateMs, record.timing.actualMs)}</dd>
              </>
            )}
          </>
        )}
        <dt>Created</dt>
        <dd>{new Date(record.createdAt).toLocaleString()}</dd>
        <dt>File</dt>
        <dd>{record.imagePath.split(/[\\/]/).pop()}</dd>
      </dl>
      {record.sourceImagePath && (
        <div className="library-panel__original">
          <span className="field-label" style={{ margin: 0 }}>
            Original
          </span>
          {sourceMissing ? (
            <p className="library-panel__warning">The original is no longer there - the kept copy was deleted.</p>
          ) : (
            <div className="library-panel__original-media">
              <ExpandButton src={window.kvgenius.imageUrlFor(record.sourceImagePath)} kind="image" filePath={record.sourceImagePath} alt="Original" />
              <img src={window.kvgenius.imageUrlFor(record.sourceImagePath)} alt="Original picture this was made from" />
            </div>
          )}
          {record.outpaint && !sourceMissing && (
            <p className="settings-hint" style={{ margin: '8px 0 0' }}>
              Extended by{' '}
              {(['left', 'top', 'right', 'bottom'] as const)
                .filter((side) => record.outpaint![side] > 0)
                .map((side) => `${record.outpaint![side]} px ${side}`)
                .join(', ')}
              . The original is untouched; only the new area was drawn.
            </p>
          )}
          {record.maskImagePath && !sourceMissing && (
            <>
              <span className="field-label" style={{ margin: '10px 0 0', display: 'block' }}>
                Mask (white was re-drawn)
              </span>
              <div className="library-panel__original-media">
                <img src={window.kvgenius.imageUrlFor(record.maskImagePath)} alt="The mask this was made with" />
              </div>
            </>
          )}
        </div>
      )}
    </>
  );

  return split ? (
    <aside ref={panelRef} className="library-panel library-panel--split">
      <div className="library-panel__image">{media}</div>
      <div className="library-panel__info">
        {header}
        {rest}
      </div>
    </aside>
  ) : (
    <aside ref={panelRef} className="library-panel">
      {header}
      {media}
      {rest}
    </aside>
  );
}
