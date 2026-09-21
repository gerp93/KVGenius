import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationKind, GenerationRecord, GenerationRef, VideoSourceRequest } from '../../shared/types';
import GeneratedVideo from '../components/GeneratedVideo';
import ExpandButton from '../components/Lightbox';
import { formatBytes } from '../utils/format';
import { justifyRows } from '../utils/justifiedRows';

const PAGE_SIZE = 60;
const TARGET_ROW_HEIGHT = 260;
const GRID_GAP = 12;
// wan22-i2v's frame rate (see Generate.tsx) - only used to show a video's length in seconds.
const VIDEO_FPS = 16;

interface Props {
  onRecall: (record: GenerationRecord) => void;
  onImageToVideo: (request: VideoSourceRequest) => void;
}

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

export default function LibraryOutput({ onRecall, onImageToVideo }: Props) {
  const [tab, setTab] = useState<GenerationKind>('image');
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [counts, setCounts] = useState<Record<GenerationKind, number>>({ image: 0, video: 0 });
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
  const [gridWidth, setGridWidth] = useState(0);

  const navigate = useNavigate();
  const gridRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  // Bumped on every tab change so a page that finishes loading for the tab we just left is
  // discarded instead of being appended to the new tab's list.
  const requestToken = useRef(0);
  // The card a Shift-click range starts from: the last one clicked in select mode.
  const anchorId = useRef<number | null>(null);
  const loadingRef = useRef(false);

  const loadPage = useCallback(async (kind: GenerationKind, beforeId: number | null, favorites: boolean, token: number) => {
    loadingRef.current = true;
    setLoading(true);
    try {
      const page = await window.kvgenius.listGenerations(kind, PAGE_SIZE, beforeId, favorites);
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
    void loadPage(tab, null, favoritesOnly, token);
  }, [tab, favoritesOnly, loadPage]);

  useEffect(() => {
    window.kvgenius
      .countGenerations(favoritesOnly)
      .then(setCounts)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [favoritesOnly]);

  const loadMore = useCallback(() => {
    // The very first page belongs to the tab-change effect above.
    if (loadingRef.current || records.length === 0) return;
    void loadPage(tab, records[records.length - 1].id, favoritesOnly, requestToken.current);
  }, [records, tab, favoritesOnly, loadPage]);

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
      else next.set(record.id, { id: record.id, imagePath: record.imagePath, favorite: record.favorite });
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
        next.set(r.id, { id: r.id, imagePath: r.imagePath, favorite: r.favorite });
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
      const refs = await window.kvgenius.listGenerationRefs(tab, favoritesOnly);
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

  async function handleDelete(record: GenerationRecord) {
    const note = record.favorite ? ' It is marked as a favorite.' : '';
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
    const note = favoriteCount > 0 ? ` ${favoriteCount} of them ${favoriteCount === 1 ? 'is a favorite' : 'are favorites'}.` : '';
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

  function renderCard(record: GenerationRecord, width: number, height: number) {
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
            {!selecting && <ExpandButton src={url} kind={isVideo ? 'video' : 'image'} filePath={record.imagePath} alt={record.prompt} />}
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
              {row.items.map(({ index, width }) => renderCard(records[index], width, row.height))}
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
              <button type="button" onClick={() => setInfoId(null)} title="Close">
                ✕
              </button>
            </span>
          </div>
          <div className="library-panel__media">
            <ExpandButton
              src={window.kvgenius.imageUrlFor(infoRecord.imagePath)}
              kind={kindOf(infoRecord) === 'video' ? 'video' : 'image'}
              filePath={infoRecord.imagePath}
              alt={infoRecord.prompt}
            />
            {kindOf(infoRecord) === 'video' ? (
              <GeneratedVideo src={window.kvgenius.imageUrlFor(infoRecord.imagePath)} filePath={infoRecord.imagePath} />
            ) : (
              <img src={window.kvgenius.imageUrlFor(infoRecord.imagePath)} alt={infoRecord.prompt} />
            )}
          </div>

          <button type="button" className="primary" onClick={() => handleRecreate(infoRecord)} style={{ width: '100%' }}>
            ↺ Re-rack
          </button>
          <div className="library-panel__actions">
            {kindOf(infoRecord) === 'image' && (
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

          <div className="field-label">Prompt</div>
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
            <dt>Created</dt>
            <dd>{new Date(infoRecord.createdAt).toLocaleString()}</dd>
            <dt>File</dt>
            <dd>{infoRecord.imagePath.split(/[\\/]/).pop()}</dd>
          </dl>
        </aside>
      )}
    </div>
  );
}
