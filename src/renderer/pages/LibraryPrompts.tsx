import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationKind, GenerationRecord, VideoSourceRequest } from '../../shared/types';
import CopyButton from '../components/CopyButton';
import CycleMedia from '../components/CycleMedia';
import GalleryLightbox from '../components/GalleryLightbox';
import DetailsDock from '../components/DetailsDock';
import OriginBadge from '../components/OriginBadge';
import LibraryDetails from '../components/LibraryDetails';
import { useBorrowedRecord } from '../hooks/useBorrowedRecord';
import { GenerationQueue } from '../hooks/useGenerationQueue';
import { useCycleIndex } from '../hooks/useCycleIndex';
import { announceGenerationChange, useGenerationChanges } from '../utils/generationChanges';
import { justifyRows, rowSizeHint } from '../utils/justifiedRows';
import CardSizeSlider from '../components/CardSizeSlider';
import { useCardScale } from '../hooks/useCardScale';
import { usePromptView } from '../hooks/usePromptView';
import { useScrollKeeper } from '../hooks/useScrollKeeper';
import { SIMILARITY, clusterPrompts, orderBySimilarity, similarityMatrix, varyingWords } from '../../shared/promptSimilarity';
import { pinNotice } from '../utils/library';
import { sourceMissingMessage } from '../../shared/sourceFamilies';
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

/** One tile of the page: a single prompt's card, or - when clustering - a fan standing for a cluster of alike prompts. */
type Tile =
  | { kind: 'prompt'; group: PinnedGroup }
  | { kind: 'cluster'; key: string; groups: PinnedGroup[]; varying: string[]; hue: number };

/** How many differing words a cluster's heading lists before "+n more". */
const VARYING_SHOWN = 8;

/**
 * The tiles to lay out. Plain: every prompt's card, in pinned order, or - ordering by likeness - with the most alike
 * side by side. Clustering: prompts that are alike collapse into one fan tile each; the rest stay single cards.
 * Ordering by likeness then also orders the tiles themselves, and the cards inside an opened cluster.
 */
function buildTiles(groups: PinnedGroup[], view: { cluster: boolean; byLikeness: boolean; threshold: number }): Tile[] {
  if (groups.length === 0) return [];
  if (!view.cluster && !view.byLikeness) return groups.map((group) => ({ kind: 'prompt', group }));

  const matrix = similarityMatrix(groups.map((g) => g.prompt));
  const arrange = (indexes: number[]) => (view.byLikeness ? orderBySimilarity(indexes, matrix) : indexes);
  if (!view.cluster) return arrange(groups.map((_, i) => i)).map((i) => ({ kind: 'prompt', group: groups[i] }));

  // A tile per cluster, led by its first-pinned member; the order of the tiles follows that member.
  const clusters = clusterPrompts(matrix, view.threshold);
  const lead = arrange(clusters.map((c) => c[0]));
  const byLead = new Map(clusters.map((c) => [c[0], c]));
  let hue = 0;
  return lead.map((first): Tile => {
    const members = byLead.get(first) ?? [first];
    if (members.length === 1) return { kind: 'prompt', group: groups[first] };
    const memberGroups = arrange(members).map((i) => groups[i]);
    const tile: Tile = {
      kind: 'cluster',
      key: `cluster-${groups[first].items[0].id}`,
      groups: memberGroups,
      varying: varyingWords(memberGroups.map((g) => g.prompt)),
      hue,
    };
    hue = (hue + 47) % 360;
    return tile;
  });
}

/** What a tile is drawn from: its leading prompt's first picture. */
function leadRecord(tile: Tile): GenerationRecord {
  return tile.kind === 'prompt' ? tile.group.items[0] : tile.groups[0].items[0];
}

/** The "differ by: red · blue · green" line for a cluster. */
function varyingText(varying: string[]): string {
  if (varying.length === 0) return '';
  const more = varying.length > VARYING_SHOWN ? ` +${varying.length - VARYING_SHOWN} more` : '';
  return `differ by: ${varying.slice(0, VARYING_SHOWN).join(' · ')}${more}`;
}

interface ClusterTileProps {
  tile: Extract<Tile, { kind: 'cluster' }>;
  width: number;
  height: number;
  onOpen: () => void;
}

/** A cluster of alike prompts as a fan of cards: the leading prompt's picture on top and the next ones peeking out
 * from behind. Clicking it opens the cluster, which then shows its prompts as ordinary cards. */
function ClusterTile({ tile, width, height, onOpen }: ClusterTileProps) {
  const lead = tile.groups[0].items[0];
  const behind = tile.groups.slice(1, 3);
  const layer = (record: GenerationRecord, className: string) => (
    <div key={`${className}-${record.id}`} className={`cluster-tile__layer ${className}`}>
      <CycleMedia paths={[record.imagePath]} index={0} isVideo={kindOf(record) === 'video'} alt={record.prompt} />
    </div>
  );
  return (
    <div className="library-card cluster-tile" style={{ width, '--cluster-hue': tile.hue } as React.CSSProperties}>
      <div className="library-card__media cluster-tile__fan" style={{ height, cursor: 'pointer' }} onClick={onOpen} title="Click to open this cluster">
        {behind.map((g, i) => layer(g.items[0], `cluster-tile__layer--back${i + 1}`)).reverse()}
        {layer(lead, 'cluster-tile__layer--top')}
        <div className="prompt-tile__prompt">{lead.prompt}</div>
      </div>
      <div className="library-card__badges">
        <span className="library-card__stack-badge" title={`${tile.groups.length} similar prompts${tile.varying.length > 0 ? ` - ${varyingText(tile.varying)}` : ''}`}>
          {tile.groups.length} similar
        </span>
      </div>
      <div className="library-card__actions library-card__actions--labeled">
        <button type="button" className="primary" onClick={onOpen} title={varyingText(tile.varying) || 'Show the prompts in this cluster'}>
          Open cluster
        </button>
      </div>
    </div>
  );
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
    <div data-card-id={group.items[0].id} className={`library-card prompt-tile${active ? ' library-card--active' : ''}`} style={{ width }}>
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
        <div className="prompt-tile__prompt">{record.prompt}</div>
      </div>
      <div className="library-card__badges">
        {group.items.length > 1 && (
          <span className="library-card__stack-badge" title={`${group.items.length} pinned pictures share this exact prompt - this card cycles through them`}>
            {index + 1} / {group.items.length}
          </span>
        )}
        <OriginBadge record={record} />
        {record.hidden && <span className="library-card__hidden-badge">Hidden</span>}
      </div>
      <div className="library-card__actions library-card__actions--labeled">
        <button type="button" className="primary" onClick={() => onUse(record)} title="Put this prompt on the Generate page">
          Use prompt
        </button>
        <button
          type="button"
          onClick={() => onRerack(record)}
          disabled={sourceMissing}
          title={sourceMissing ? sourceMissingMessage(record.modelFamily) : "Load this picture's prompt and settings (size, seed, steps)"}
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
  const [cardScale, setCardScale] = useCardScale();
  const [view, setView] = usePromptView();
  // The cluster that was opened: the page then shows just its prompts, as ordinary cards.
  const [openKey, setOpenKey] = useState<string | null>(null);
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
  // The scrolling column around the grid: kept in place when cards leave it or the details panel opens and closes.
  const scrollerRef = useRef<HTMLElement | null>(null);
  scrollerRef.current = gridRef.current?.closest<HTMLElement>('.library-output__main') ?? null;
  const keepView = useScrollKeeper(scrollerRef);
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
  const tiles = useMemo(
    () => buildTiles(visibleGroups, view),
    // Only the choices that change the layout: the threshold means nothing while not clustering.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [visibleGroups, view.cluster, view.byLikeness, view.cluster ? view.threshold : 0]
  );
  const openTile = tiles.find((t): t is Extract<Tile, { kind: 'cluster' }> => t.kind === 'cluster' && t.key === openKey) ?? null;
  // What is on show: the tiles, or - inside an opened cluster - just its prompts.
  const shown = useMemo<Tile[]>(() => (openTile ? openTile.groups.map((group) => ({ kind: 'prompt', group })) : tiles), [tiles, openTile]);
  // Every picture on show, in card order: what the full-window viewer steps through.
  const visible = useMemo(
    () => shown.flatMap((t) => (t.kind === 'prompt' ? t.group.items : t.groups.flatMap((g) => g.items))),
    [shown]
  );

  // Each card is as big as its first picture; the others in the group are fitted into that box.
  const rows = justifyRows(
    shown.map((t) => ({ aspect: leadRecord(t).width / Math.max(leadRecord(t).height, 1) })),
    gridWidth,
    TARGET_ROW_HEIGHT * cardScale,
    GRID_GAP
  );
  const clusterCount = tiles.filter((t) => t.kind === 'cluster').length;

  function openCluster(key: string | null) {
    setOpenKey(key);
    scrollerRef.current?.scrollTo({ top: 0 });
  }

  // The original of a result opened from its panel may not be pinned (so not in this list): it is shown from here instead.
  const borrow = useBorrowedRecord();
  const infoRecord = records.find((r) => r.id === infoId) ?? (borrow.borrowed && borrow.borrowed.id === infoId ? borrow.borrowed : null);

  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));

  /** Takes a generation out of the list (unpinned, hidden, deleted), closing whatever showed it. */
  function forget(id: number) {
    keepView.hold([id]);
    setRecords((prev) => prev.filter((r) => r.id !== id));
    setInfoId((prev) => (prev === id ? null : prev));
    // The viewer was showing one of these by index - close it rather than land on a shifted item.
    setLightboxIndex(null);
  }

  function patch(id: number, changes: Partial<GenerationRecord>) {
    setRecords((prev) => prev.map((r) => (r.id === id ? { ...r, ...changes } : r)));
    borrow.patch(id, changes);
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

  /** Extending a video: Generate opens with its last frame as the source image. */
  function handleExtendVideo(request: VideoSourceRequest) {
    onImageToVideo(request);
    navigate('/');
  }

  function handleImageToImage(record: GenerationRecord) {
    onImageToVideo({ target: 'image', imagePath: record.imagePath, width: record.width, height: record.height, from: record });
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
            {view.cluster && clusterCount > 0 ? ` in ${clusterCount} cluster${clusterCount === 1 ? '' : 's'}` : ''}
            {visible.length !== visibleGroups.length ? ` - ${visible.length} pictures` : ''}
          </span>
          <button
            type="button"
            className={view.cluster ? 'primary' : undefined}
            onClick={() => {
              setOpenKey(null);
              setView({ cluster: !view.cluster });
            }}
            title="Gather prompts that are mostly the same words - say, one or two changed - into one fan of cards"
          >
            Cluster
          </button>
          {view.cluster && (
            <label className="prompt-library__similarity" title="How much of the wording two prompts must share to be in one cluster. Lower = bigger, looser clusters.">
              <span>Alike</span>
              <input
                type="range"
                min={SIMILARITY.min}
                max={SIMILARITY.max}
                step={SIMILARITY.step}
                value={view.threshold}
                aria-label="How alike prompts must be to cluster"
                onChange={(e) => setView({ threshold: Number(e.target.value) })}
                onDoubleClick={() => setView({ threshold: SIMILARITY.default })}
              />
              <span className="prompt-library__similarity-value">{Math.round(view.threshold * 100)}%</span>
            </label>
          )}
          <button
            type="button"
            className={view.byLikeness ? 'primary' : undefined}
            onClick={() => setView({ byLikeness: !view.byLikeness })}
            title="Put the most alike prompts next to each other, instead of in the order they were pinned"
          >
            Order by likeness
          </button>
          <span style={{ flex: 1 }} />
          <CardSizeSlider scale={cardScale} onChange={setCardScale} />
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

        {view.cluster && visibleGroups.length > 1 && clusterCount === 0 && (
          <p style={{ color: 'var(--color-text-muted)' }}>
            No prompts are {Math.round(view.threshold * 100)}% alike - lower the Alike slider to cluster looser matches.
          </p>
        )}

        {openTile && (
          <div className="prompt-cluster__heading" style={{ '--cluster-hue': openTile.hue } as React.CSSProperties}>
            <button type="button" onClick={() => openCluster(null)} title="Back to all the prompts">
              ← All prompts
            </button>
            <strong>{openTile.groups.length} similar prompts</strong>
            {openTile.varying.length > 0 && (
              <span className="prompt-cluster__varying" title={openTile.varying.join(', ')}>
                {varyingText(openTile.varying)}
              </span>
            )}
          </div>
        )}

        {/* Always rendered so the width observer attaches on mount. */}
        <div className="library-rows" ref={gridRef}>
          {rows.map((row) => (
            <div key={leadRecord(shown[row.items[0].index]).id} className="library-row library-row--skippable" style={rowSizeHint(row.height)}>
              {row.items.map(({ index, width }) => {
                const tile = shown[index];
                if (tile.kind === 'cluster') {
                  return <ClusterTile key={tile.key} tile={tile} width={width} height={row.height} onOpen={() => openCluster(tile.key)} />;
                }
                const group = tile.group;
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
            onClose={() => {
              keepView.hold();
              setInfoId(null);
            }}
            onOpenRecord={(target) => {
              borrow.borrow(target);
              setInfoId(target.id);
            }}
            onExpand={() => {
              const index = visible.findIndex((r) => r.id === infoRecord.id);
              if (index >= 0) setLightboxIndex(index);
              else borrow.setExpanded(true);
            }}
            onToggleFavorite={handleToggleFavorite}
            onTogglePinned={handleTogglePinned}
            onToggleHidden={handleToggleHidden}
            onDelete={handleDelete}
            onRerack={handleRerack}
            onImageToVideo={handleImageToVideo}
            onImageToImage={handleImageToImage}
            onExtendVideo={handleExtendVideo}
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

      {borrow.expanded && infoRecord && (
        <GalleryLightbox
          src={window.kvgenius.imageUrlFor(infoRecord.imagePath)}
          kind={kindOf(infoRecord)}
          filePath={infoRecord.imagePath}
          alt={infoRecord.prompt}
          hasPrev={false}
          hasNext={false}
          onPrev={() => undefined}
          onNext={() => undefined}
          onClose={() => borrow.setExpanded(false)}
        />
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
