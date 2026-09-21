import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationKind, GenerationRecord, VideoSourceRequest } from '../../shared/types';
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
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [infoId, setInfoId] = useState<number | null>(null);
  const [gridWidth, setGridWidth] = useState(0);

  const navigate = useNavigate();
  const gridRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  // Bumped on every tab change so a page that finishes loading for the tab we just left is
  // discarded instead of being appended to the new tab's list.
  const requestToken = useRef(0);
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
    setSelectedIds(new Set());
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
  // area of the scrolling `.page`. The observer is rebuilt after every load so it re-reports
  // "still visible" and keeps filling a tall window without needing a scroll event.
  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore || loading) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) loadMore();
      },
      { root: sentinel.closest('.page'), rootMargin: '0px 0px 800px 0px' }
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

  function handleTabChange(next: GenerationKind) {
    if (next !== tab) setTab(next);
  }

  function toggleSelected(id: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function exitSelectMode() {
    setSelecting(false);
    setSelectedIds(new Set());
  }

  function handleCardClick(record: GenerationRecord) {
    // In select mode a card only toggles its selection; otherwise it opens the info panel.
    if (selecting) toggleSelected(record.id);
    else setInfoId(record.id);
  }

  function handleRecreate(record: GenerationRecord) {
    onRecall(record);
    navigate('/');
  }

  function handleImageToVideo(record: GenerationRecord) {
    onImageToVideo({ imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  function forgetRecords(deleted: GenerationRecord[]) {
    const ids = new Set(deleted.map((r) => r.id));
    setRecords((prev) => prev.filter((r) => !ids.has(r.id)));
    setCounts((prev) => {
      const next = { ...prev };
      for (const r of deleted) next[kindOf(r)] = Math.max(0, next[kindOf(r)] - 1);
      return next;
    });
    setInfoId((prev) => (prev !== null && ids.has(prev) ? null : prev));
  }

  async function handleToggleFavorite(record: GenerationRecord) {
    const favorite = !record.favorite;
    try {
      await window.kvgenius.setGenerationFavorite(record.id, favorite);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    if (favoritesOnly && !favorite) {
      // Un-favoriting from the favorites view: it no longer belongs in this list.
      forgetRecords([record]);
    } else {
      setRecords((prev) => prev.map((r) => (r.id === record.id ? { ...r, favorite } : r)));
    }
  }

  async function handleDelete(record: GenerationRecord) {
    const note = record.favorite ? ' It is marked as a favorite.' : '';
    if (!window.confirm(`Delete this generation? This removes the file from disk too.${note}`)) return;
    try {
      await window.kvgenius.deleteGeneration(record.id, record.imagePath);
      forgetRecords([record]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteSelected() {
    const toDelete = records.filter((r) => selectedIds.has(r.id));
    if (toDelete.length === 0) return;
    const favoriteCount = toDelete.filter((r) => r.favorite).length;
    const note = favoriteCount > 0 ? ` ${favoriteCount} of them ${favoriteCount === 1 ? 'is a favorite' : 'are favorites'}.` : '';
    if (!window.confirm(`Delete ${toDelete.length} generation${toDelete.length === 1 ? '' : 's'}? This removes the files from disk too.${note}`)) {
      return;
    }
    try {
      await Promise.all(toDelete.map((r) => window.kvgenius.deleteGeneration(r.id, r.imagePath)));
      forgetRecords(toDelete);
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
    const selected = selecting && selectedIds.has(record.id);
    const active = !selecting && infoId === record.id;
    return (
      <div
        key={record.id}
        className={`library-card${selected ? ' library-card--selected' : ''}${active ? ' library-card--active' : ''}`}
        style={{ width }}
      >
        {selecting && (
          <span className="library-card__select">
            <input type="checkbox" checked={selectedIds.has(record.id)} readOnly tabIndex={-1} />
          </span>
        )}
        <div onClick={() => handleCardClick(record)} style={{ cursor: 'pointer' }}>
          <div className="library-card__media" style={{ height }}>
            {isVideo ? (
              <>
                {/* First frame as the thumbnail: the #t fragment seeks just past 0 so the browser
                    paints a frame instead of a blank box. */}
                <video src={`${url}#t=0.1`} preload="metadata" muted playsInline />
                <span className="library-card__play-badge">▶</span>
              </>
            ) : (
              <img src={url} alt={record.prompt} loading="lazy" decoding="async" />
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

        <div className="button-row" style={{ marginBottom: 12 }}>
          <button type="button" className={tab === 'image' ? 'primary' : undefined} onClick={() => handleTabChange('image')}>
            🖼️ Images ({counts.image})
          </button>
          <button type="button" className={tab === 'video' ? 'primary' : undefined} onClick={() => handleTabChange('video')}>
            🎬 Videos ({counts.video})
          </button>
          <button
            type="button"
            className={favoritesOnly ? 'primary' : undefined}
            onClick={() => setFavoritesOnly((v) => !v)}
            title="Show only favorites"
          >
            {favoritesOnly ? '★' : '☆'} Favorites
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

        {records.length > 0 && !selecting && (
          <div className="button-row" style={{ marginBottom: 12 }}>
            <button type="button" onClick={() => setSelecting(true)}>
              Delete Multiple
            </button>
          </div>
        )}

        {records.length > 0 && selecting && (
          <div className="button-row" style={{ marginBottom: 12 }}>
            <button
              type="button"
              onClick={() => setSelectedIds(new Set(records.map((r) => r.id)))}
              disabled={selectedIds.size === records.length}
            >
              {hasMore ? 'Select All Loaded' : 'Select All'}
            </button>
            <button type="button" onClick={() => setSelectedIds(new Set())} disabled={selectedIds.size === 0}>
              Clear Selection
            </button>
            <button type="button" onClick={handleDeleteSelected} disabled={selectedIds.size === 0}>
              Delete Selected ({selectedIds.size})
            </button>
            <button type="button" onClick={exitSelectMode}>
              Cancel
            </button>
          </div>
        )}

        <div className="library-rows" ref={gridRef}>
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
            {kindOf(infoRecord) === 'video' ? (
              <video src={window.kvgenius.imageUrlFor(infoRecord.imagePath)} controls preload="metadata" />
            ) : (
              <img src={window.kvgenius.imageUrlFor(infoRecord.imagePath)} alt={infoRecord.prompt} />
            )}
          </div>

          <button type="button" className="primary" onClick={() => handleRecreate(infoRecord)} style={{ width: '100%' }}>
            ↺ Recreate in Generate
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
            <dt>Size</dt>
            <dd>
              {infoRecord.width} × {infoRecord.height}
            </dd>
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
