import { useEffect, useRef, useState } from 'react';
import { MAX_PENDING_JOBS } from '../hooks/useGenerationQueue';
import type { GenerationQueue, Job } from '../hooks/useGenerationQueue';
import ImageDropZone from '../components/ImageDropZone';
import { DEFAULT_UPSCALE_FACTOR, UPSCALE_FACTORS, UPSCALE_FAMILY, fileNameOf, nearestUpscaleFactor, upscaledSize } from '../../shared/upscale';
import type { UpscaleRecall } from '../../shared/upscale';

const MODEL_KEY = 'kvgenius-tools-upscale-model';
const FACTOR_KEY = 'kvgenius-tools-upscale-factor';
/** How many of this session's upscales the results list shows. */
const RESULTS_SHOWN = 12;

interface Picked {
  path: string;
  /** The picture's own size, known once its thumbnail has loaded. */
  size?: { width: number; height: number };
}

function readSaved(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Not remembered - the page still works.
  }
}

function savedFactor(): number {
  const value = Number(readSaved(FACTOR_KEY));
  return UPSCALE_FACTORS.includes(value) ? value : DEFAULT_UPSCALE_FACTOR;
}

function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
}

const STATUS_LABEL: Record<Job['status'], string> = {
  queued: 'Waiting',
  running: 'Upscaling...',
  done: 'Done',
  failed: 'Failed',
};

interface Props {
  queue: GenerationQueue;
  /** Open the app-wide queue bar (upscales were just queued). */
  onShowQueue: () => void;
  /** A kept original sent here to be upscaled again (a re-rack, or Library > Sources); null otherwise. */
  recall: UpscaleRecall | null;
  /** Called once the page has taken the recall, so it is not taken again. */
  onRecallHandled: () => void;
}

/**
 * Tools > Upscale: enlarge any pictures, not only ones already in the Library. Pick files, choose a
 * model and a size, and each becomes a job in the shared queue (so it runs in line with everything
 * else, with the same progress and timing). Finished upscales are saved to Library > Output as new
 * images, and are listed here for the session.
 */
export default function ToolsUpscale({ queue, onShowQueue, recall, onRecallHandled }: Props) {
  const [picked, setPicked] = useState<Picked[]>([]);
  // For each picture sent here to be re-run: the width its earlier result came out at, so the size
  // choice can be set to match once the picture's own size is known.
  const recallWidths = useRef<Map<string, number>>(new Map());
  const [models, setModels] = useState<string[] | null>(null);
  const [model, setModelState] = useState(() => readSaved(MODEL_KEY) ?? '');
  const [factor, setFactorState] = useState(savedFactor);
  const [unreachable, setUnreachable] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  function setModel(next: string) {
    setModelState(next);
    save(MODEL_KEY, next);
  }

  function setFactor(next: number) {
    setFactorState(next);
    save(FACTOR_KEY, String(next));
  }

  /** Asks ComfyUI which upscale models it has (the list only exists while it is running). */
  function loadModels(isCancelled: () => boolean = () => false) {
    setUnreachable(false);
    window.kvgenius
      .listUpscaleModels()
      .then((list) => {
        if (isCancelled()) return;
        setModels(list);
        setModelState((prev) => (list.includes(prev) ? prev : (list[0] ?? '')));
      })
      .catch(() => {
        if (!isCancelled()) setUnreachable(true);
      });
  }

  useEffect(() => {
    let cancelled = false;
    loadModels(() => cancelled);
    return () => {
      cancelled = true;
    };
  }, []);

  /** Adds pictures to the list (from the file dialog or a drop), skipping any already there. */
  function addPaths(paths: string[]) {
    if (paths.length === 0) return;
    setError(null);
    setNotice(null);
    setPicked((prev) => {
      const have = new Set(prev.map((p) => p.path));
      return [...prev, ...paths.filter((p) => !have.has(p)).map((path) => ({ path }))];
    });
  }

  async function handleChoose() {
    setError(null);
    try {
      addPaths(await window.kvgenius.chooseSourceImages());
    } catch (err) {
      setError(cleanError(err));
    }
  }

  function handleLoaded(path: string, img: HTMLImageElement) {
    const size = { width: img.naturalWidth, height: img.naturalHeight };
    setPicked((prev) => prev.map((p) => (p.path === path ? { ...p, size } : p)));
    const earlierWidth = recallWidths.current.get(path);
    if (earlierWidth) {
      recallWidths.current.delete(path);
      setFactor(nearestUpscaleFactor(size.width, earlierWidth));
    }
  }

  // An upscale re-racked from the Library or the queue, or a picture sent from Library > Sources:
  // its kept original is put on the list, at the size it was made at.
  useEffect(() => {
    if (!recall) return;
    if (recall.outputWidth > 0) recallWidths.current.set(recall.sourcePath, recall.outputWidth);
    addPaths([recall.sourcePath]);
    onRecallHandled();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recall]);

  function handleUpscale() {
    if (!model) return;
    const ready = picked.filter((p) => p.size);
    if (ready.length === 0) return;
    setError(null);
    const added = queue.enqueue(
      ready.map((p) => ({
        family: UPSCALE_FAMILY,
        kind: 'image' as const,
        params: {
          prompt: `Upscale of ${fileNameOf(p.path)}`,
          ...upscaledSize(p.size!.width, p.size!.height, factor),
          seed: 0,
          steps: 1,
          cfg: 1,
          sourceImagePath: p.path,
          upscaleModel: model,
        },
      }))
    );
    if (added === 0) {
      setError(`The queue is full (${MAX_PENDING_JOBS} waiting).`);
      return;
    }
    onShowQueue();
    // What was left over (the queue was nearly full) stays picked so it can be sent again.
    const sent = new Set(ready.slice(0, added).map((p) => p.path));
    setPicked((prev) => prev.filter((p) => !sent.has(p.path)));
    setNotice(
      `Queued ${added} upscale${added === 1 ? '' : 's'}. Progress is in the queue bar; each result is also saved to Library > Output.` +
        (added < ready.length ? ` ${ready.length - added} did not fit in the queue and are still picked.` : '')
    );
  }

  async function runFileAction(action: () => Promise<unknown>) {
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  const ready = picked.filter((p) => p.size).length;
  const results = queue.jobs
    .filter((job) => job.family === UPSCALE_FAMILY && !job.dismissed)
    .slice()
    .reverse()
    .slice(0, RESULTS_SHOWN);

  return (
    <ImageDropZone className="tools-upscale" multiple onPaths={addPaths} onReject={setError}>
      <h2 className="tools-upscale__title">Upscale</h2>
      <p className="tools-upscale__hint">
        Enlarge pictures with an AI upscale model. Drop images anywhere on this page, or choose them; results are saved to Library &gt;
        Output as new images, and the originals are left alone.
      </p>

      <div className="tools-upscale__controls">
        <button type="button" onClick={() => void handleChoose()}>
          Choose images...
        </button>
        <select value={model} onChange={(e) => setModel(e.target.value)} title="Upscale model" disabled={unreachable}>
          {models === null && !unreachable && <option value="">Loading models...</option>}
          {unreachable && <option value="">ComfyUI is not reachable</option>}
          {models?.length === 0 && <option value="">No upscale models installed</option>}
          {models?.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <select value={factor} onChange={(e) => setFactor(Number(e.target.value))} title="Size multiplier">
          {UPSCALE_FACTORS.map((f) => (
            <option key={f} value={f}>
              {f}x
            </option>
          ))}
        </select>
        <button type="button" className="primary" onClick={handleUpscale} disabled={!model || ready === 0 || unreachable}>
          {ready > 1 ? `Upscale ${ready} images` : 'Upscale'}
        </button>
        {unreachable && (
          <button type="button" onClick={() => loadModels()}>
            Retry
          </button>
        )}
      </div>

      {error && <p className="tools-upscale__error">{error}</p>}
      {notice && <p className="tools-upscale__notice">{notice}</p>}

      {picked.length > 0 && (
        <div className="tools-upscale__picked">
          {picked.map((p) => {
            const out = p.size ? upscaledSize(p.size.width, p.size.height, factor) : null;
            return (
              <div key={p.path} className="tools-upscale__card">
                <img
                  src={window.kvgenius.imageUrlFor(p.path)}
                  alt={fileNameOf(p.path)}
                  onLoad={(e) => handleLoaded(p.path, e.currentTarget)}
                  onError={() => setError(`Could not read ${fileNameOf(p.path)} as an image.`)}
                />
                <div className="tools-upscale__card-text">
                  <strong title={p.path}>{fileNameOf(p.path)}</strong>
                  <span>
                    {p.size && out ? `${p.size.width} × ${p.size.height}  →  ${out.width} × ${out.height}` : 'Reading size...'}
                  </span>
                </div>
                <button
                  type="button"
                  className="tools-upscale__remove"
                  title="Remove"
                  onClick={() => setPicked((prev) => prev.filter((x) => x.path !== p.path))}
                >
                  ✕
                </button>
              </div>
            );
          })}
        </div>
      )}

      {results.length > 0 && (
        <>
          <h3 className="tools-upscale__subtitle">This session</h3>
          <div className="tools-upscale__results">
            {results.map((job) => (
              <div key={job.id} className="tools-upscale__card">
                {job.status === 'done' && job.imageUrl ? (
                  <img src={job.imageUrl} alt={job.record?.prompt ?? 'Upscaled image'} />
                ) : (
                  <div className="tools-upscale__placeholder">{STATUS_LABEL[job.status]}</div>
                )}
                <div className="tools-upscale__card-text">
                  <strong>{job.record ? `${job.record.width} × ${job.record.height}` : `${job.params.width} × ${job.params.height}`}</strong>
                  <span title={job.error}>{job.status === 'failed' ? (job.error ?? 'Failed') : STATUS_LABEL[job.status]}</span>
                </div>
                {job.status === 'done' && job.record && (
                  <div className="tools-upscale__card-actions">
                    <button
                      type="button"
                      onClick={() => void runFileAction(() => window.kvgenius.saveGenerationAs(job.record!.imagePath))}
                    >
                      Save as...
                    </button>
                    <button
                      type="button"
                      onClick={() => void runFileAction(() => window.kvgenius.revealGenerationInFileManager(job.record!.imagePath))}
                    >
                      Show in folder
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </ImageDropZone>
  );
}
