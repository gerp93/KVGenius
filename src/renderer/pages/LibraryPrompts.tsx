import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationKind, GenerationRecord } from '../../shared/types';
import GalleryLightbox from '../components/GalleryLightbox';
import GeneratedVideo from '../components/GeneratedVideo';
import { justifyRows } from '../utils/justifiedRows';

const TARGET_ROW_HEIGHT = 240;
const GRID_GAP = 12;

interface Props {
  onRecallPrompt: (prompt: string) => void;
  onRecall: (record: GenerationRecord) => void;
}

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

/**
 * The pinned generations: one picture per "look", chosen in the Library or on the Generate page.
 * There is no separate saved prompt - a tile's prompt is just the prompt of the generation shown.
 */
export default function LibraryPrompts({ onRecallPrompt, onRecall }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  // Hidden items (see Settings > Hidden Content) are left out unless this is on, as in Library > Output.
  const [showHidden, setShowHidden] = useState(false);
  const [gridWidth, setGridWidth] = useState(0);
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
  }, [showHidden]);

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

  function handleUse(record: GenerationRecord) {
    onRecallPrompt(record.prompt);
    navigate('/');
  }

  /** Loads the picture's own settings too (size, seed, steps), not just its prompt. */
  function handleRerack(record: GenerationRecord) {
    onRecall(record);
    navigate('/');
  }

  async function handleUnpin(record: GenerationRecord) {
    try {
      await window.kvgenius.setGenerationPinned(record.id, false);
      setRecords((prev) => prev.filter((r) => r.id !== record.id));
      // The viewer was showing one of these by index - close it rather than land on a shifted item.
      setLightboxIndex(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function renderTile(record: GenerationRecord, index: number, width: number, height: number) {
    const url = window.kvgenius.imageUrlFor(record.imagePath);
    return (
      <div key={record.id} className="library-card prompt-tile" style={{ width }}>
        <div
          className="library-card__media"
          style={{ height, cursor: 'pointer' }}
          onClick={() => handleUse(record)}
          title="Click to use this prompt"
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
              void handleUnpin(record);
            }}
          >
            📌
          </button>
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
    <div className="prompt-library">
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
        <button
          type="button"
          className={showHidden ? 'primary' : undefined}
          onClick={() => setShowHidden((v) => !v)}
          title="Include pinned items hidden by the hidden-words rule or by hand"
        >
          {showHidden ? '🙈 Showing hidden' : '🙈 Show hidden'}
        </button>
      </div>

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
