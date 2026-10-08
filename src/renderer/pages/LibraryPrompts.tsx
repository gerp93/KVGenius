import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationKind, GenerationRecord, VideoSourceRequest } from '../../shared/types';
import CopyButton from '../components/CopyButton';
import CycleMedia from '../components/CycleMedia';
import GalleryLightbox from '../components/GalleryLightbox';
import DetailsDock from '../components/DetailsDock';
import OriginBadge from '../components/OriginBadge';
import LibraryDetails from '../components/LibraryDetails';
import { GenerationQueue } from '../hooks/useGenerationQueue';
import { useCycleIndex } from '../hooks/useCycleIndex';
import { announceGenerationChange, useGenerationChanges } from '../utils/generationChanges';
import { justifyRows } from '../utils/justifiedRows';
import { pinNotice } from '../utils/library';
import { SOURCE_MISSING_MESSAGE } from '../../shared/sourceFamilies';
import { useSourceMissing } from '../hooks/useMissingSources';

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
  /** Open the app-wide queue panel (an upscale was just queued). */
  onShowQueue: () => void;
}

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

/** The pinned pictures that share one exact prompt - they make one card, which cycles through them. */
interface PinnedGroup {
  prompt: string;
  items: GenerationRecord[];
}

/** What makes two pins the same picture: with every one of these equal the output would be identical. */
function settingsKey(r: GenerationRecord): string {
  return [r.modelFamily, r.width, r.height, r.seed, r.steps, r.cfg, r.length].join('|');
}

/**
 * Pinned items grouped by exact prompt (a video and an image with the same prompt stay apart), in the
 * order the first of each group was pinned. A pin whose settings are identical to one already in its group
 * is the same picture, so it joins the group without being shown twice.
 */
function groupPinned(records: GenerationRecord[]): PinnedGroup[] {
  const groups = new Map<string, PinnedGroup>();
  const seen = new Set<string>();
  for (const record of records) {
    const key = `${kindOf(record)}\u0000${record.prompt}`;
    const picture = `${key}\u0000${settingsKey(record)}`;
    if (seen.has(picture)) continue;
    seen.add(picture);
    const group = groups.get(key);
    if (group) group.items.push(record);
    else groups.set(key, { prompt: record.prompt, items: [record] });
  }
  return [...groups.values()];
}

interface TileProps {
  group: PinnedGroup;
  width: number;
  height: number;
  active: boolean;
  onOpen: (record: GenerationRecord) => void;
  onExpand: (record: GenerationRecord) => void;
  onUnpin: (record: GenerationRecord) => void;
  onUse: (record: GenerationRecord) => void;
  onRerack: (record: GenerationRecord) => void;
}

/** One card of the Prompts gallery. With several pinned pictures it cycles through them (resting while the
 * pointer is over it), and everything on it - open, expand, unpin, copy, re-rack - applies to the one on show. */
function PinnedTile({ group, width, height, active, onOpen, onExpand, onUnpin, onUse, onRerack }: TileProps) {
  const [hovered, setHovered] = useState(false);
  const index = useCycleIndex(group.items.length, hovered);
  const record = group.items[index] ?? group.items[0];
  const isVideo = kindOf(record) === 'video';
  const sourceMissing = useSourceMissing(record);
  return (
    <div className={`library-card prompt-tile${active ? ' library-card--active' : ''}`} style={{ width }}>
      <div
        className="library-card__media"
        style={{ height, cursor: 'pointer' }}
        onClick={() => onOpen(record)}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        title="Click for details"
      >
        <CycleMedia paths={group.items.map((r) => r.imagePath)} index={index} isVideo={isVideo} alt={record.prompt} />
        {isVideo && <span className="library-card__play-badge">▶</span>}
        <button
          type="button"
          className="expand-button"
          title="Expand"
          onClick={(e) => {
            e.stopPropagation();
            onExpand(record);
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
            onUnpin(record);
          }}
        >
          📌
        </button>
        <CopyButton compact className="prompt-tile__copy" text={record.prompt} title="Copy this prompt" />
        {!isVideo && <CopyButton compact className="prompt-tile__copy prompt-tile__copy--image" imagePath={record.imagePath} title="Copy the image" />}
        <div className="library-card__badges">
          {group.items.length > 1 && (
            <span className="library-card__stack-badge" title={`${group.items.length} pinned pictures share this exact prompt - this card cycles through them`}>
              {index + 1} / {group.items.length}
            </span>
          )}
          <OriginBadge record={record} />
          {record.hidden && <span className="library-card__hidden-badge">Hidden</span>}
        </div>
        <div className="prompt-tile__prompt">{record.prompt}</div>
      </div>
      <div className="library-card__actions library-card__actions--labeled">
        <button type="button" className="primary" onClick={() => onUse(record)} title="Put this prompt on the Generate page">
          Use prompt
        </button>
        <button
          type="button"
          onClick={() => onRerack(record)}
          disabled={sourceMissing}
          title={sourceMissing ? SOURCE_MISSING_MESSAGE : "Load this picture's prompt and settings (size, seed, steps)"}
        >
          ↺ Re-rack
        </button>
      </div>
    </div>
  );
}

/**
 * The pinned generations, one card per exact prompt, chosen in the Library or on the Generate page.
 * There is no separate saved prompt - a card's prompt is just the prompt of the picture shown. Pins that
 * share a prompt share a card, which cycles through their pictures. Clicking a card opens the same
 * details panel as Library > Output.
 */
export default function LibraryPrompts({ queue, onRecallPrompt, onRecall, onImageToVideo, showHidden, onShowQueue }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [gridWidth, setGridWidth] = useState(0);
  const [infoId, setInfoId] = useState<number | null>(null);
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

  const groups = useMemo(() => groupPinned(records), [records]);
  const visibleGroups = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return needle ? groups.filter((g) => g.prompt.toLowerCase().includes(needle)) : groups;
  }, [groups, search]);
  // Every picture on show, in card order: what the full-window viewer steps through.
  const visible = useMemo(() => visibleGroups.flatMap((g) => g.items), [visibleGroups]);

  // Each card is as big as its first picture; the others in the group are fitted into that box.
  const rows = justifyRows(
    visibleGroups.map((g) => ({ aspect: g.items[0].width / Math.max(g.items[0].height, 1) })),
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

  // Changed from the queue bar: bring this list in line with it.
  useGenerationChanges((change) => {
    switch (change.kind) {
      case 'favorite':
        patch(change.id, { favorite: change.favorite, imagePath: change.imagePath });
        break;
      case 'pinned':
      case 'hidden':
        // A pin or unpin (or a hide) changes which pictures belong here at all: load the list again.
        setReloadKey((k) => k + 1);
        break;
      case 'trashed':
        forget(change.id);
        break;
      case 'queueDetailsOpened':
        // Only one details panel at a time: the app-wide one just opened.
        setInfoId(null);
        break;
    }
  });
  useEffect(() => {
    if (infoId !== null) announceGenerationChange({ kind: 'libraryDetailsOpened' });
  }, [infoId]);

  function handleUse(record: GenerationRecord) {
    onRecallPrompt(record.prompt);
    navigate('/');
  }

  /** Loads the picture's own settings too (size, seed, steps), not just its prompt. */
  function handleRerack(record: GenerationRecord) {
    // The app opens the right page for it: Generate, or Tools > Upscale for an upscale.
    onRecall(record);
  }

  function handleImageToVideo(record: GenerationRecord) {
    onImageToVideo({ imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  function handleImageToImage(record: GenerationRecord) {
    onImageToVideo({ target: 'image', imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  async function handleTogglePinned(record: GenerationRecord) {
    const pinned = !record.pinned;
    let groupSize: number;
    try {
      ({ groupSize } = await window.kvgenius.setGenerationPinned(record.id, pinned));
    } catch (err) {
      fail(err);
      return;
    }
    // Unpinned: it no longer belongs on this page (the image stays in Library > Output).
    if (!pinned) forget(record.id);
    else {
      patch(record.id, { pinned });
      setNotice(pinNotice(groupSize));
    }
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
            {visibleGroups.length === groups.length
              ? `${groups.length} prompt${groups.length === 1 ? '' : 's'}`
              : `${visibleGroups.length} of ${groups.length}`}
            {visible.length !== visibleGroups.length ? ` - ${visible.length} pictures` : ''}
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
            <div key={visibleGroups[row.items[0].index].items[0].id} className="library-row">
              {row.items.map(({ index, width }) => {
                const group = visibleGroups[index];
                return (
                  <PinnedTile
                    key={group.items[0].id}
                    group={group}
                    width={width}
                    height={row.height}
                    active={group.items.some((r) => r.id === infoId)}
                    onOpen={(record) => setInfoId(record.id)}
                    onExpand={(record) => setLightboxIndex(visible.findIndex((r) => r.id === record.id))}
                    onUnpin={handleTogglePinned}
                    onUse={handleUse}
                    onRerack={handleRerack}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {infoRecord && (
        <DetailsDock>
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
            onImageToImage={handleImageToImage}
            onExtendVideo={onImageToVideo}
            onSaveAs={handleSaveAs}
            onReveal={handleReveal}
            onUpscaleQueued={onShowQueue}
            // A GIF is a new image, not a pinned one: it lives in Library > Output.
            onGifMade={() => undefined}
            onError={setError}
            onNotice={setNotice}
          />
        </DetailsDock>
      )}

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
