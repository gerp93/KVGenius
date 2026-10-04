import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DEFAULT_GIF_FPS, DEFAULT_GIF_WIDTH, GIF_FPS_CHOICES, GIF_WIDTHS } from '../../shared/gif';
import { UPSCALE_FACTORS, DEFAULT_UPSCALE_FACTOR, UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY, isUpscaleFamily } from '../../shared/upscale';
import { FAMILY_KIND, GenerationKind, GenerationRecord, GenerationRef, VideoSourceRequest } from '../../shared/types';
import GeneratedVideo from '../components/GeneratedVideo';
import QueuePanel from '../components/QueuePanel';
import { GenerationQueue, MAX_PENDING_JOBS } from '../hooks/useGenerationQueue';
import GalleryLightbox from '../components/GalleryLightbox';
import { formatBytes, formatDifference, formatDuration } from '../utils/format';
import { justifyRows } from '../utils/justifiedRows';

const PAGE_SIZE = 60;
const TARGET_ROW_HEIGHT = 260;
const GRID_GAP = 12;
// wan22-i2v's frame rate (see Generate.tsx) - only used to show a video's length in seconds.
const VIDEO_FPS = 16;

interface Props {
  queue: GenerationQueue;
  onRecall: (record: GenerationRecord) => void;
  onImageToVideo: (request: VideoSourceRequest) => void;
}

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

/** GIFs made from a video are stored as images, but can't be upscaled or animated again. */
function isGif(record: GenerationRecord): boolean {
  return record.imagePath.toLowerCase().endsWith('.gif');
}

export default function LibraryOutput({ queue, onRecall, onImageToVideo }: Props) {
  const [tab, setTab] = useState<GenerationKind>('image');
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  // Hidden items (see Settings > Hidden Content) are left out of the Library unless this is on.
  const [showHidden, setShowHidden] = useState(false);
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [counts, setCounts] = useState<Record<GenerationKind, number>>({ image: 0, video: 0 });
  // Image tab only: show just one file type (e.g. 'gif'); null = all. `extensions` are the types present.
  const [extension, setExtension] = useState<string | null>(null);
  const [extensions, setExtensions] = useState<string[]>([]);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selecting, setSelecting] = useState(false);
  // Selected generations by id. Holds just what's needed to delete/export, so Select All can
  // cover pages that aren't loaded yet.
  const [selection, setSelection] = useState<Map<number, GenerationRef>>(new Map());
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [infoSize, setInfoSize] = useState<number | null>(null);
  const [infoId, setInfoId] = useState<number | null>(null);
  // Upscale controls in the details panel: models come from ComfyUI the first time they are needed.
  const [upscaleModels, setUpscaleModels] = useState<string[] | null>(null);
  const [upscaleModel, setUpscaleModel] = useState('');
  const [upscaleFactor, setUpscaleFactor] = useState(DEFAULT_UPSCALE_FACTOR);
  const [queueCollapsed, setQueueCollapsed] = useState(false);
  // Upscale jobs already merged into the list below (those finished before this page opened are in its load).
  const mergedUpscales = useRef<Set<number> | null>(null);
  // GIF conversion controls for a video's details panel.
  const [gifWidth, setGifWidth] = useState(DEFAULT_GIF_WIDTH);
  const [gifFps, setGifFps] = useState(DEFAULT_GIF_FPS);
  const [makingGif, setMakingGif] = useState(false);
  const [gridWidth, setGridWidth] = useState(0);
  // Index into `records` of the image/video open in the full-window gallery viewer, if any.
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const navigate = useNavigate();
  const gridRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  // Bumped on every tab change so a page that finishes loading for the tab we just left is
  // discarded instead of being appended to the new tab's list.
  const requestToken = useRef(0);
  // The card a Shift-click range starts from: the last one clicked in select mode.
  const anchorId = useRef<number | null>(null);
  const loadingRef = useRef(false);

  const loadPage = useCallback(async (kind: GenerationKind, beforeId: number | null, favorites: boolean, hidden: boolean, ext: string | null, token: number) => {
    loadingRef.current = true;
    setLoading(true);
    try {
      const page = await window.kvgenius.listGenerations(kind, PAGE_SIZE, beforeId, favorites, hidden, kind === 'image' ? ext : null);
      if (token !== requestToken.current) return;
      setRecords((prev) => (beforeId === null ? page : [...prev, ...page]));
      setHasMore(page.length === PAGE_SIZE);
    } catch (err) {
      if (token !== requestToken.current) return;
      setError(err instanceof Error ? err.message : String(err));
      setHasMore(false);
    } finally {
      if (token === requestToken.current) {
        loadingRef.current = false;
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const token = ++requestToken.current;
    loadingRef.current = false;
    setRecords([]);
    setHasMore(true);
    setSelection(new Map());
    anchorId.current = null;
    setInfoId(null);
    setLightboxIndex(null);
    void loadPage(tab, null, favoritesOnly, showHidden, extension, token);
  }, [tab, favoritesOnly, showHidden, extension, loadPage]);

  useEffect(() => {
    window.kvgenius
      .countGenerations(favoritesOnly, showHidden, extension)
      .then(setCounts)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [favoritesOnly, showHidden, extension]);

  // The file types offered in the filter. A type whose last image was deleted drops out of the list,
  // and the filter goes back to "all" if it was set to that type.
  const refreshExtensions = useCallback(() => {
    window.kvgenius
      .listImageExtensions()
      .then((list) => {
        setExtensions(list);
        setExtension((prev) => (prev && !list.includes(prev) ? null : prev));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshExtensions();
  }, [refreshExtensions, counts.image]);

  /** Whether a new image belongs in the list as filtered: its type has to be the chosen one. */
  const matchesExtension = (record: GenerationRecord) =>
    !extension || record.imagePath.toLowerCase().endsWith(`.${extension}`);

  const loadMore = useCallback(() => {
    // The very first page belongs to the tab-change effect above.
    if (loadingRef.current || records.length === 0) return;
    void loadPage(tab, records[records.length - 1].id, favoritesOnly, showHidden, extension, requestToken.current);
  }, [records, tab, favoritesOnly, showHidden, extension, loadPage]);

  // Infinite scroll: load the next page when the sentinel below the grid gets near the visible
  // area of the scrolling grid column. The observer is rebuilt after every load so it re-reports
  // "still visible" and keeps filling a tall window without needing a scroll event.
  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore || loading) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) loadMore();
      },
      { root: sentinel.closest('.library-output__main'), rootMargin: '0px 0px 800px 0px' }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMore, loading, loadMore]);

  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setGridWidth(el.clientWidth));
    observer.observe(el);
    setGridWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  const rows = justifyRows(
    records.map((r) => ({ aspect: r.width / Math.max(r.height, 1) })),
    gridWidth,
    TARGET_ROW_HEIGHT,
    GRID_GAP
  );
  const infoRecord = records.find((r) => r.id === infoId) ?? null;

  const infoPath = infoRecord?.imagePath ?? null;
  useEffect(() => {
    setInfoSize(null);
    if (!infoPath) return;
    let cancelled = false;
    window.kvgenius
      .getFileSize(infoPath)
      .then((size) => {
        if (!cancelled) setInfoSize(size);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [infoPath]);

  function handleTabChange(next: GenerationKind) {
    if (next !== tab) setTab(next);
  }

  function toggleSelected(record: GenerationRecord) {
    setSelection((prev) => {
      const next = new Map(prev);
      if (next.has(record.id)) next.delete(record.id);
      else next.set(record.id, { id: record.id, imagePath: record.imagePath, favorite: record.favorite, pinned: record.pinned });
      return next;
    });
  }

  /** Shift-click: selects every card from the last one clicked to this one, in the order they are
   * shown. Cards already selected stay selected; nothing is deselected. */
  function selectRangeTo(record: GenerationRecord) {
    const from = records.findIndex((r) => r.id === anchorId.current);
    const to = records.findIndex((r) => r.id === record.id);
    if (from < 0 || to < 0) {
      toggleSelected(record);
      anchorId.current = record.id;
      return;
    }
    const [lo, hi] = from < to ? [from, to] : [to, from];
    setSelection((prev) => {
      const next = new Map(prev);
      for (const r of records.slice(lo, hi + 1)) {
        next.set(r.id, { id: r.id, imagePath: r.imagePath, favorite: r.favorite, pinned: r.pinned });
      }
      return next;
    });
  }

  function exitSelectMode() {
    setSelecting(false);
    setSelection(new Map());
    setNotice(null);
    anchorId.current = null;
  }

  /** Selects every generation matching the current tab and filter - not just the loaded ones. */
  async function handleSelectAll() {
    setBusy(true);
    try {
      const refs = await window.kvgenius.listGenerationRefs(tab, favoritesOnly, showHidden, tab === 'image' ? extension : null);
      setSelection(new Map(refs.map((ref) => [ref.id, ref])));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleExportSelected() {
    const refs = [...selection.values()];
    if (refs.length === 0) return;
    setBusy(true);
    setNotice(null);
    try {
      const result = await window.kvgenius.exportGenerations(refs.map((r) => r.imagePath));
      if (result.status === 'saved') {
        setNotice(`Exported ${result.count} file${result.count === 1 ? '' : 's'} to ${result.path}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function handleCardClick(record: GenerationRecord, event: React.MouseEvent) {
    if (!selecting) {
      setInfoId(record.id);
      return;
    }
    // Select mode: a click toggles that card (so Ctrl/Cmd-click works the same), and Shift-click
    // selects the whole run from the last card clicked. The last plain click is the range's anchor.
    if (event.shiftKey && anchorId.current !== null) {
      selectRangeTo(record);
    } else {
      toggleSelected(record);
      anchorId.current = record.id;
    }
  }

  /** Pins (or unpins) a generation as the example of its prompt - the Library > Prompts gallery. */
  async function handleTogglePinned(record: GenerationRecord) {
    const pinned = !record.pinned;
    try {
      await window.kvgenius.setGenerationPinned(record.id, pinned);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    setRecords((prev) => prev.map((r) => (r.id === record.id ? { ...r, pinned } : r)));
    setSelection((prev) => {
      const ref = prev.get(record.id);
      return ref ? new Map(prev).set(record.id, { ...ref, pinned }) : prev;
    });
    setNotice(pinned ? 'Pinned - find it under Library > Prompts.' : null);
  }

  function handleRecreate(record: GenerationRecord) {
    onRecall(record);
    navigate('/');
  }

  function handleImageToVideo(record: GenerationRecord) {
    onImageToVideo({ imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  function forgetIds(ids: number[], kind: GenerationKind) {
    const gone = new Set(ids);
    setRecords((prev) => prev.filter((r) => !gone.has(r.id)));
    setCounts((prev) => ({ ...prev, [kind]: Math.max(0, prev[kind] - ids.length) }));
    setInfoId((prev) => (prev !== null && gone.has(prev) ? null : prev));
    // The gallery viewer was showing one of these by index - close it rather than have it land
    // on a now-shifted, unrelated item.
    setLightboxIndex((prev) => (prev !== null && gone.has(records[prev]?.id) ? null : prev));
  }

  async function handleToggleFavorite(record: GenerationRecord) {
    const favorite = !record.favorite;
    let imagePath: string;
    try {
      // Favoriting moves the file into the favorites folder (and back), so its path can change.
      ({ imagePath } = await window.kvgenius.setGenerationFavorite(record.id, favorite));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    if (favoritesOnly && !favorite) {
      // Un-favoriting from the favorites view: it no longer belongs in this list.
      forgetIds([record.id], kindOf(record));
    } else {
      setRecords((prev) => prev.map((r) => (r.id === record.id ? { ...r, favorite, imagePath } : r)));
    }
    setSelection((prev) => {
      const ref = prev.get(record.id);
      return ref ? new Map(prev).set(record.id, { ...ref, imagePath, favorite }) : prev;
    });
  }

  async function loadUpscaleModels() {
    try {
      const models = await window.kvgenius.listUpscaleModels();
      setUpscaleModels(models);
      setUpscaleModel((prev) => (models.includes(prev) ? prev : (models[0] ?? '')));
    } catch {
      setError('Could not reach ComfyUI to list upscale models.');
    }
  }

  // Fetch the model list the first time an item's details are opened.
  const infoOpen = infoRecord !== null;
  useEffect(() => {
    if (infoOpen && upscaleModels === null) void loadUpscaleModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [infoOpen, upscaleModels]);

  /** Adds an upscale job to the shared queue; several can be waiting at once. */
  function handleUpscale(record: GenerationRecord) {
    if (!upscaleModel) return;
    setNotice(null);
    setError(null);
    setQueueCollapsed(false);
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
    if (added === 0) setError(`The queue is full (${MAX_PENDING_JOBS} waiting).`);
  }

  // A finished upscale is a new image: put it at the top of the list without a reload.
  useEffect(() => {
    if (mergedUpscales.current === null) {
      mergedUpscales.current = new Set(queue.jobs.filter((j) => isUpscaleFamily(j.family) && j.status === 'done').map((j) => j.id));
      return;
    }
    const merged = mergedUpscales.current;
    for (const job of queue.jobs) {
      if (!isUpscaleFamily(job.family) || job.status !== 'done' || !job.record || merged.has(job.id)) continue;
      merged.add(job.id);
      const made = job.record;
      if (made.hidden && !showHidden) continue;
      if (favoritesOnly) setNotice('An upscale finished - it is not a favorite, so turn off the Favorites filter to see it.');
      const fits = job.kind !== 'image' || matchesExtension(made);
      if (tab === job.kind && !favoritesOnly && fits) {
        setRecords((prev) => (prev.some((r) => r.id === made.id) ? prev : [made, ...prev]));
      }
      if (fits) setCounts((prev) => ({ ...prev, [job.kind]: prev[job.kind] + 1 }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue.jobs, tab, favoritesOnly, showHidden, extension]);

  // The queue panel pops out while anything is running or waiting, and stays for finished/failed upscales.
  const showQueue = queue.jobs.some(
    (j) =>
      j.status === 'queued' ||
      j.status === 'running' ||
      (isUpscaleFamily(j.family) && (j.status === 'done' || (j.status === 'failed' && !j.dismissed)))
  );

  async function handleMakeGif(record: GenerationRecord) {
    setMakingGif(true);
    setNotice(null);
    setError(null);
    try {
      const { record: made } = await window.kvgenius.convertToGif(record.id, { fps: gifFps, width: gifWidth });
      setNotice(`Made a ${made.width} × ${made.height} GIF - saved as a new image.`);
      const fits = matchesExtension(made) && (!made.hidden || showHidden);
      if (tab === 'image' && !favoritesOnly && fits) {
        setRecords((prev) => [made, ...prev]);
      }
      if (fits) setCounts((prev) => ({ ...prev, image: prev.image + 1 }));
      refreshExtensions();
    } catch (err) {
      // Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method".
      const message = err instanceof Error ? err.message : String(err);
      setError(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''));
    } finally {
      setMakingGif(false);
    }
  }

  async function handleToggleHidden(record: GenerationRecord) {
    const hidden = !record.hidden;
    try {
      await window.kvgenius.setGenerationHidden(record.id, hidden);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    if (hidden && !showHidden) {
      // Hiding while hidden items are filtered out: it no longer belongs in this list.
      forgetIds([record.id], kindOf(record));
    } else {
      setRecords((prev) => prev.map((r) => (r.id === record.id ? { ...r, hidden } : r)));
    }
  }

  async function handleDelete(record: GenerationRecord) {
    const note = (record.favorite ? ' It is marked as a favorite.' : '') + (record.pinned ? ' It is pinned under Prompts.' : '');
    if (!window.confirm(`Delete this generation? This removes the file from disk too.${note}`)) return;
    try {
      await window.kvgenius.deleteGeneration(record.id, record.imagePath);
      forgetIds([record.id], kindOf(record));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteSelected() {
    const toDelete = [...selection.values()];
    if (toDelete.length === 0) return;
    const favoriteCount = toDelete.filter((r) => r.favorite).length;
    const pinnedCount = toDelete.filter((r) => r.pinned).length;
    const note =
      (favoriteCount > 0 ? ` ${favoriteCount} of them ${favoriteCount === 1 ? 'is a favorite' : 'are favorites'}.` : '') +
      (pinnedCount > 0 ? ` ${pinnedCount} of them ${pinnedCount === 1 ? 'is' : 'are'} pinned under Prompts.` : '');
    if (!window.confirm(`Delete ${toDelete.length} generation${toDelete.length === 1 ? '' : 's'}? This removes the files from disk too.${note}`)) {
      return;
    }
    try {
      await Promise.all(toDelete.map((r) => window.kvgenius.deleteGeneration(r.id, r.imagePath)));
      forgetIds(toDelete.map((r) => r.id), tab);
      exitSelectMode();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSaveAs(record: GenerationRecord) {
    try {
      await window.kvgenius.saveGenerationAs(record.imagePath);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleReveal(record: GenerationRecord) {
    try {
      await window.kvgenius.revealGenerationInFileManager(record.imagePath);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function renderCard(record: GenerationRecord, index: number, width: number, height: number) {
    const url = window.kvgenius.imageUrlFor(record.imagePath);
    const isVideo = kindOf(record) === 'video';
    const selected = selecting && selection.has(record.id);
    const active = !selecting && infoId === record.id;
    return (
      <div
        key={record.id}
        className={`library-card${selected ? ' library-card--selected' : ''}${active ? ' library-card--active' : ''}`}
        style={{ width }}
      >
        {selecting && (
          <span className="library-card__select">
            <input type="checkbox" checked={selection.has(record.id)} readOnly tabIndex={-1} />
          </span>
        )}
        <div onClick={(e) => handleCardClick(record, e)} style={{ cursor: 'pointer' }}>
          <div className="library-card__media" style={{ height }}>
            {isVideo ? (
              <>
                <GeneratedVideo src={url} filePath={record.imagePath} thumbnail />
                <span className="library-card__play-badge">▶</span>
              </>
            ) : (
              <img src={url} alt={record.prompt} loading="lazy" decoding="async" />
            )}
            {!selecting && (
              <button
                type="button"
                className="expand-button"
                title="Expand"
                onClick={(e) => {
                  e.stopPropagation();
                  setLightboxIndex(index);
                }}
              >
                ⤢
              </button>
            )}
            {record.hidden && <span className="library-card__hidden-badge">Hidden</span>}
            {record.pinned && (
              <span className="library-card__pinned-badge" title="Pinned under Prompts">
                📌
              </span>
            )}
            {!selecting && (
              <button
                type="button"
                className={`library-card__fav${record.favorite ? ' library-card__fav--on' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  void handleToggleFavorite(record);
                }}
                title={record.favorite ? 'Remove from favorites' : 'Add to favorites'}
              >
                {record.favorite ? '★' : '☆'}
              </button>
            )}
          </div>
          <div className="library-card__info" title={record.prompt}>
            {record.prompt}
          </div>
        </div>
        {!selecting && (
          <div className="library-card__actions">
            <button type="button" onClick={() => setInfoId(record.id)} title="Info">
              ℹ️
            </button>
            {!isVideo && (
              <button type="button" onClick={() => handleImageToVideo(record)} title="Create video from image">
                🎬
              </button>
            )}
            <button type="button" onClick={() => handleSaveAs(record)} title="Save As...">
              💾
            </button>
            <button type="button" onClick={() => handleReveal(record)} title="Show in File Manager">
              📂
            </button>
            <button type="button" onClick={() => handleDelete(record)} title="Delete">
              🗑️
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="library-output">
      <div className="library-output__main">
        {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

        <div className="library-toolbar">
          {selecting ? (
            <>
              <span className="library-toolbar__hint">Click to select, Shift-click for a range</span>
              <button
                type="button"
                onClick={handleSelectAll}
                disabled={busy || counts[tab] === 0 || selection.size === counts[tab]}
                title="Select every item in this tab, including ones not scrolled into view yet"
              >
                Select All ({counts[tab]})
              </button>
              <button type="button" onClick={() => setSelection(new Map())} disabled={busy || selection.size === 0}>
                Clear Selection
              </button>
              <button type="button" onClick={handleExportSelected} disabled={busy || selection.size === 0}>
                Export Selected ({selection.size})
              </button>
              <button type="button" onClick={handleDeleteSelected} disabled={busy || selection.size === 0}>
                Delete Selected ({selection.size})
              </button>
              <button type="button" onClick={exitSelectMode}>
                Cancel
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className={favoritesOnly ? 'primary' : undefined}
                onClick={() => setFavoritesOnly((v) => !v)}
                title="Show only favorites"
              >
                {favoritesOnly ? '★' : '☆'} Favorites
              </button>
              <button
                type="button"
                className={showHidden ? 'primary' : undefined}
                onClick={() => setShowHidden((v) => !v)}
                title="Include items hidden by the hidden-words rule or by hand"
              >
                {showHidden ? '🙈 Showing hidden' : '🙈 Show hidden'}
              </button>
              {tab === 'image' && extensions.length > 0 && (
                <select
                  value={extension ?? ''}
                  onChange={(e) => setExtension(e.target.value || null)}
                  title="Show only one file type"
                >
                  <option value="">All types</option>
                  {extensions.map((ext) => (
                    <option key={ext} value={ext}>
                      {ext.toUpperCase()}
                    </option>
                  ))}
                </select>
              )}
              <button type="button" onClick={() => setSelecting(true)} disabled={records.length === 0}>
                Select Multiple
              </button>
            </>
          )}
        </div>

        {notice && <p className="library-notice">{notice}</p>}

        <div className="tab-strip" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'image'}
            className={`tab-strip__tab${tab === 'image' ? ' active' : ''}`}
            onClick={() => handleTabChange('image')}
          >
            🖼️ Images ({counts.image})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'video'}
            className={`tab-strip__tab${tab === 'video' ? ' active' : ''}`}
            onClick={() => handleTabChange('video')}
          >
            🎬 Videos ({counts.video})
          </button>
        </div>

        {!loading && records.length === 0 && (
          <p style={{ color: 'var(--color-text-muted)' }}>
            {favoritesOnly
              ? `No favorite ${tab === 'video' ? 'videos' : 'images'} yet - tap ☆ on one to save it here.`
              : tab === 'video'
                ? 'No videos yet - go make something.'
                : 'No images yet - go make something.'}
          </p>
        )}

        {/* user-select is off in select mode so Shift-click picks a range instead of highlighting text */}
        <div className={`library-rows${selecting ? ' library-rows--selecting' : ''}`} ref={gridRef}>
          {rows.map((row) => (
            <div key={records[row.items[0].index].id} className="library-row">
              {row.items.map(({ index, width }) => renderCard(records[index], index, width, row.height))}
            </div>
          ))}
        </div>

        <div ref={sentinelRef} className="library-sentinel">
          {loading && records.length > 0 && 'Loading more...'}
        </div>
      </div>

      {infoRecord && (
        <aside className="library-panel">
          <div className="library-panel__header">
            <strong>Details</strong>
            <span style={{ display: 'flex', gap: 6 }}>
              <button type="button" onClick={() => handleToggleFavorite(infoRecord)}>
                {infoRecord.favorite ? '★ Favorited' : '☆ Favorite'}
              </button>
              <button
                type="button"
                onClick={() => handleTogglePinned(infoRecord)}
                title={
                  infoRecord.pinned
                    ? 'Unpin - remove this from Library > Prompts'
                    : 'Pin as the example of this prompt, shown under Library > Prompts'
                }
              >
                {infoRecord.pinned ? '📌 Pinned' : '📌 Pin'}
              </button>
              <button type="button" onClick={() => handleToggleHidden(infoRecord)}>
                {infoRecord.hidden ? 'Unhide' : 'Hide'}
              </button>
              <button type="button" onClick={() => setInfoId(null)} title="Close">
                ✕
              </button>
            </span>
          </div>
          <div className="library-panel__media">
            <button
              type="button"
              className="expand-button"
              title="Expand"
              onClick={() => setLightboxIndex(records.findIndex((r) => r.id === infoRecord.id))}
            >
              ⤢
            </button>
            {kindOf(infoRecord) === 'video' ? (
              <GeneratedVideo src={window.kvgenius.imageUrlFor(infoRecord.imagePath)} filePath={infoRecord.imagePath} />
            ) : (
              <img src={window.kvgenius.imageUrlFor(infoRecord.imagePath)} alt={infoRecord.prompt} />
            )}
          </div>

          <button type="button" className="primary" onClick={() => handleRecreate(infoRecord)} style={{ width: '100%' }}>
            ↺ Re-rack
          </button>
          {kindOf(infoRecord) === 'video' && (
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
                <button type="button" onClick={() => handleMakeGif(infoRecord)} disabled={makingGif}>
                  {makingGif ? 'Converting...' : 'Make GIF'}
                </button>
              </div>
            </div>
          )}
          {!isGif(infoRecord) && (
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
                <select
                  value={upscaleFactor}
                  onChange={(e) => setUpscaleFactor(Number(e.target.value))}
                  title="Size multiplier"
                >
                  {UPSCALE_FACTORS.map((f) => (
                    <option key={f} value={f}>
                      {f}×
                    </option>
                  ))}
                </select>
                <button type="button" onClick={() => handleUpscale(infoRecord)} disabled={!upscaleModel}>
                  Upscale
                </button>
              </div>
            </div>
          )}
          <div className="library-panel__actions">
            {kindOf(infoRecord) === 'image' && !isGif(infoRecord) && (
              <button type="button" onClick={() => handleImageToVideo(infoRecord)} title="Create video from image">
                🎬 Video
              </button>
            )}
            <button type="button" onClick={() => handleSaveAs(infoRecord)} title="Save As...">
              💾
            </button>
            <button type="button" onClick={() => handleReveal(infoRecord)} title="Show in File Manager">
              📂
            </button>
            <button type="button" onClick={() => handleDelete(infoRecord)} title="Delete">
              🗑️
            </button>
          </div>

          <div className="library-panel__prompt-header">
            <span className="field-label" style={{ margin: 0 }}>
              Prompt
            </span>
          </div>
          <p className="library-panel__prompt">{infoRecord.prompt}</p>

          <dl className="library-panel__meta">
            <dt>Type</dt>
            <dd>{kindOf(infoRecord) === 'video' ? 'Video' : 'Image'}</dd>
            <dt>Model</dt>
            <dd>{infoRecord.modelFamily}</dd>
            <dt>Dimensions</dt>
            <dd>
              {infoRecord.width} × {infoRecord.height}
            </dd>
            <dt>File size</dt>
            <dd>{infoSize === null ? '-' : formatBytes(infoSize)}</dd>
            {infoRecord.length !== null && (
              <>
                <dt>Length</dt>
                <dd>
                  {Math.round(((infoRecord.length - 1) / VIDEO_FPS) * 4) / 4}s ({infoRecord.length} frames)
                </dd>
              </>
            )}
            <dt>Seed</dt>
            <dd>{infoRecord.seed}</dd>
            {kindOf(infoRecord) === 'image' && (
              <>
                <dt>Steps</dt>
                <dd>{infoRecord.steps}</dd>
                <dt>CFG</dt>
                <dd>{infoRecord.cfg}</dd>
              </>
            )}
            {infoRecord.timing && (
              <>
                <dt>Estimated</dt>
                <dd>{infoRecord.timing.estimateMs === null ? 'no estimate yet' : formatDuration(infoRecord.timing.estimateMs)}</dd>
                <dt>Took</dt>
                <dd>
                  {formatDuration(infoRecord.timing.actualMs)}
                  {infoRecord.timing.loadMs !== null && infoRecord.timing.loadMs >= 2000
                    ? ` (${formatDuration(infoRecord.timing.loadMs)} loading models)`
                    : ''}
                </dd>
                {infoRecord.timing.estimateMs !== null && (
                  <>
                    <dt>Difference</dt>
                    <dd>{formatDifference(infoRecord.timing.estimateMs, infoRecord.timing.actualMs)}</dd>
                  </>
                )}
              </>
            )}
            <dt>Created</dt>
            <dd>{new Date(infoRecord.createdAt).toLocaleString()}</dd>
            <dt>File</dt>
            <dd>{infoRecord.imagePath.split(/[\\/]/).pop()}</dd>
          </dl>
        </aside>
      )}

      {showQueue && (
        <div className={`library-queue${queueCollapsed ? ' library-queue--collapsed' : ''}`}>
          <QueuePanel
            jobs={queue.jobs}
            now={queue.now}
            progressInfo={queue.progressInfo}
            collapsed={queueCollapsed}
            onToggle={() => setQueueCollapsed((v) => !v)}
            onCancelJob={queue.cancelJob}
            onClearQueued={queue.clearQueued}
            onDismissFailed={queue.dismissFailed}
            onToggleFavorite={handleToggleFavorite}
            onRerack={(record) => {
              onRecall(record);
              navigate('/');
            }}
          />
        </div>
      )}

      {lightboxIndex !== null && records[lightboxIndex] && (
        <GalleryLightbox
          src={window.kvgenius.imageUrlFor(records[lightboxIndex].imagePath)}
          kind={kindOf(records[lightboxIndex])}
          filePath={records[lightboxIndex].imagePath}
          alt={records[lightboxIndex].prompt}
          hasPrev={lightboxIndex > 0}
          hasNext={lightboxIndex < records.length - 1}
          onPrev={() => setLightboxIndex((i) => (i !== null ? Math.max(0, i - 1) : i))}
          onNext={() => setLightboxIndex((i) => (i !== null ? Math.min(records.length - 1, i + 1) : i))}
          onClose={() => setLightboxIndex(null)}
        />
      )}
    </div>
  );
}
