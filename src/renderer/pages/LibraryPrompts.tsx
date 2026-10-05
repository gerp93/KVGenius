import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationKind, GenerationRecord, VideoSourceRequest } from '../../shared/types';
import CopyButton from '../components/CopyButton';
import GalleryLightbox from '../components/GalleryLightbox';
import GeneratedVideo from '../components/GeneratedVideo';
import LibraryDetails from '../components/LibraryDetails';
import LibraryQueue from '../components/LibraryQueue';
import { GenerationQueue } from '../hooks/useGenerationQueue';
import { justifyRows } from '../utils/justifiedRows';

const TARGET_ROW_HEIGHT = 240;
const GRID_GAP = 12;

interface Props {
  queue: GenerationQueue;
  /** Put just a prompt's text on the Generate page (a new tab). */
  onRecallPrompt: (prompt: string) => void;
  /** Load a whole generation - prompt and settings - into the Generate page (a new tab). */
  onRecall: (record: GenerationRecord) => void;
  onImageToVideo: (request: VideoSourceRequest) => void;
  /** The app-wide "Show hidden" switch (top bar). Hidden items (Settings > Hidden Content) are left out unless it is on. */
  showHidden: boolean;
}

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

/**
 * The pinned generations: one picture per "look", chosen in the Library or on the Generate page.
 * There is no separate saved prompt - a tile's prompt is just the prompt of the generation shown.
 * Clicking a tile opens the same details panel as Library > Output.
 */
export default function LibraryPrompts({ queue, onRecallPrompt, onRecall, onImageToVideo, showHidden }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [gridWidth, setGridWidth] = useState(0);
  const [infoId, setInfoId] = useState<number | null>(null);
  const [queueCollapsed, setQueueCollapsed] = useState(false);
  // The item the last Delete moved to the Trash, so its notice can offer to put it back; and a counter
  // that makes the list load again after that.
  const [trashedId, setTrashedId] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listPinnedGenerations(showHidden)
      .then((list) => {
        if (cancelled) return;
        setRecords(list);
        setLoaded(true);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [showHidden, reloadKey]);

  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setGridWidth(el.clientWidth));
    observer.observe(el);
    setGridWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return needle ? records.filter((r) => r.prompt.toLowerCase().includes(needle)) : records;
  }, [records, search]);

  const rows = justifyRows(
    visible.map((r) => ({ aspect: r.width / Math.max(r.height, 1) })),
    gridWidth,
    TARGET_ROW_HEIGHT,
    GRID_GAP
  );
  const infoRecord = records.find((r) => r.id === infoId) ?? null;

  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));

  /** Takes a generation out of the list (unpinned, hidden, deleted), closing whatever showed it. */
  function forget(id: number) {
    setRecords((prev) => prev.filter((r) => r.id !== id));
    setInfoId((prev) => (prev === id ? null : prev));
    // The viewer was showing one of these by index - close it rather than land on a shifted item.
    setLightboxIndex(null);
  }

  function patch(id: number, changes: Partial<GenerationRecord>) {
    setRecords((prev) => prev.map((r) => (r.id === id ? { ...r, ...changes } : r)));
  }

  function handleUse(record: GenerationRecord) {
    onRecallPrompt(record.prompt);
    navigate('/');
  }

  /** Loads the picture's own settings too (size, seed, steps), not just its prompt. */
  function handleRerack(record: GenerationRecord) {
    onRecall(record);
    navigate('/');
  }

  function handleImageToVideo(record: GenerationRecord) {
    onImageToVideo({ imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  async function handleTogglePinned(record: GenerationRecord) {
    const pinned = !record.pinned;
    try {
      await window.kvgenius.setGenerationPinned(record.id, pinned);
    } catch (err) {
      fail(err);
      return;
    }
    // Unpinned: it no longer belongs on this page (the image stays in Library > Output).
    if (!pinned) forget(record.id);
    else patch(record.id, { pinned });
  }

  async function handleToggleFavorite(record: GenerationRecord) {
    const favorite = !record.favorite;
    try {
      // Favoriting moves the file into the favorites folder (and back), so its path can change.
      const { imagePath } = await window.kvgenius.setGenerationFavorite(record.id, favorite);
      patch(record.id, { favorite, imagePath });
    } catch (err) {
      fail(err);
    }
  }

  async function handleToggleHidden(record: GenerationRecord) {
    const hidden = !record.hidden;
    try {
      await window.kvgenius.setGenerationHidden(record.id, hidden);
    } catch (err) {
      fail(err);
      return;
    }
    // Hiding while hidden items are filtered out: it no longer belongs in this list.
    if (hidden && !showHidden) forget(record.id);
    else patch(record.id, { hidden });
  }

  // Delete moves to the Trash with no confirmation: undo it here, or restore it from Library > Trash.
  async function handleDelete(record: GenerationRecord) {
    try {
      const result = await window.kvgenius.trashGenerations([record.id], { includeKept: true });
      if (result.moved === 0) {
        setError('Could not move it to the Trash.');
        return;
      }
      forget(record.id);
      setTrashedId(record.id);
    } catch (err) {
      fail(err);
    }
  }

  async function handleUndoTrash() {
    if (trashedId === null) return;
    const id = trashedId;
    setTrashedId(null);
    try {
      await window.kvgenius.restoreGenerations([id]);
      setReloadKey((k) => k + 1);
    } catch (err) {
      fail(err);
    }
  }

  // The undo offer lapses after a while.
  useEffect(() => {
    if (trashedId === null) return;
    const timer = setTimeout(() => setTrashedId(null), 15000);
    return () => clearTimeout(timer);
  }, [trashedId]);

  async function handleSaveAs(record: GenerationRecord) {
    try {
      await window.kvgenius.saveGenerationAs(record.imagePath);
    } catch (err) {
      fail(err);
    }
  }

  async function handleReveal(record: GenerationRecord) {
    try {
      await window.kvgenius.revealGenerationInFileManager(record.imagePath);
    } catch (err) {
      fail(err);
    }
  }

  function renderTile(record: GenerationRecord, index: number, width: number, height: number) {
    const url = window.kvgenius.imageUrlFor(record.imagePath);
    return (
      <div key={record.id} className={`library-card prompt-tile${infoId === record.id ? ' library-card--active' : ''}`} style={{ width }}>
        <div
          className="library-card__media"
          style={{ height, cursor: 'pointer' }}
          onClick={() => setInfoId(record.id)}
          title="Click for details"
        >
          {kindOf(record) === 'video' ? (
            <>
              <GeneratedVideo src={url} filePath={record.imagePath} thumbnail />
              <span className="library-card__play-badge">▶</span>
            </>
          ) : (
            <img src={url} alt={record.prompt} loading="lazy" decoding="async" />
          )}
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
          <button
            type="button"
            className="library-card__fav library-card__fav--on"
            title="Unpin - the image stays in the Library"
            onClick={(e) => {
              e.stopPropagation();
              void handleTogglePinned(record);
            }}
          >
            📌
          </button>
          <CopyButton compact className="prompt-tile__copy" text={record.prompt} title="Copy this prompt" />
          {record.hidden && <span className="library-card__hidden-badge">Hidden</span>}
          <div className="prompt-tile__prompt">{record.prompt}</div>
        </div>
        <div className="library-card__actions">
          <button type="button" className="primary" onClick={() => handleUse(record)} title="Put this prompt on the Generate page">
            Use prompt
          </button>
          <button type="button" onClick={() => handleRerack(record)} title="Load this picture's prompt and settings (size, seed, steps)">
            ↺ Re-rack
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="library-output">
      <div className="library-output__main prompt-library">
        {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

        <div className="prompt-library__toolbar">
          <input
            type="text"
            className="prompt-library__search"
            value={search}
            placeholder="Search prompts..."
            onChange={(e) => setSearch(e.target.value)}
          />
          <span className="prompt-library__count">
            {visible.length === records.length ? `${records.length} pinned` : `${visible.length} of ${records.length}`}
          </span>
        </div>

        {notice && <p className="library-notice">{notice}</p>}
        {trashedId !== null && (
          <p className="library-notice">
            Moved to the Trash.{' '}
            <button type="button" onClick={handleUndoTrash}>
              ↶ Undo
            </button>
          </p>
        )}

        {loaded && records.length === 0 && (
          <p style={{ color: 'var(--color-text-muted)' }}>
            Nothing pinned yet - tap 📌 Pin on an image in Library &gt; Output, or on a result on the Generate page, to keep it
            here as the example of its prompt.
          </p>
        )}
        {loaded && records.length > 0 && visible.length === 0 && (
          <p style={{ color: 'var(--color-text-muted)' }}>No pinned prompts match.</p>
        )}

        {/* Always rendered so the width observer attaches on mount. */}
        <div className="library-rows" ref={gridRef}>
          {rows.map((row) => (
            <div key={visible[row.items[0].index].id} className="library-row">
              {row.items.map(({ index, width }) => renderTile(visible[index], index, width, row.height))}
            </div>
          ))}
        </div>
      </div>

      {infoRecord && (
        <LibraryDetails
          record={infoRecord}
          queue={queue}
          onClose={() => setInfoId(null)}
          onExpand={() => setLightboxIndex(visible.findIndex((r) => r.id === infoRecord.id))}
          onToggleFavorite={handleToggleFavorite}
          onTogglePinned={handleTogglePinned}
          onToggleHidden={handleToggleHidden}
          onDelete={handleDelete}
          onRerack={handleRerack}
          onImageToVideo={handleImageToVideo}
          onSaveAs={handleSaveAs}
          onReveal={handleReveal}
          onUpscaleQueued={() => setQueueCollapsed(false)}
          // A GIF is a new image, not a pinned one: it lives in Library > Output.
          onGifMade={() => undefined}
          onError={setError}
          onNotice={setNotice}
        />
      )}

      <LibraryQueue
        queue={queue}
        collapsed={queueCollapsed}
        onToggle={() => setQueueCollapsed((v) => !v)}
        onToggleFavorite={handleToggleFavorite}
        onRerack={handleRerack}
      />

      {lightboxIndex !== null && visible[lightboxIndex] && (
        <GalleryLightbox
          src={window.kvgenius.imageUrlFor(visible[lightboxIndex].imagePath)}
          kind={kindOf(visible[lightboxIndex])}
          filePath={visible[lightboxIndex].imagePath}
          alt={visible[lightboxIndex].prompt}
          hasPrev={lightboxIndex > 0}
          hasNext={lightboxIndex < visible.length - 1}
          onPrev={() => setLightboxIndex((i) => (i !== null ? Math.max(0, i - 1) : i))}
          onNext={() => setLightboxIndex((i) => (i !== null ? Math.min(visible.length - 1, i + 1) : i))}
          onClose={() => setLightboxIndex(null)}
        />
      )}
    </div>
  );
}
