import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { isUpscaleFamily } from '../../shared/upscale';
import { FAMILY_KIND, GenerationKind, GenerationRecord, GenerationRef, LibraryListOptions, VideoSourceRequest } from '../../shared/types';
import CompareOverlay from '../components/CompareOverlay';
import CopyButton from '../components/CopyButton';
import GeneratedVideo from '../components/GeneratedVideo';
import LibraryDetails from '../components/LibraryDetails';
import LibraryQueue from '../components/LibraryQueue';
import { GenerationQueue } from '../hooks/useGenerationQueue';
import GalleryLightbox from '../components/GalleryLightbox';
import { justifyRows } from '../utils/justifiedRows';

const PAGE_SIZE = 60;
const TARGET_ROW_HEIGHT = 260;
const GRID_GAP = 12;

interface Props {
  queue: GenerationQueue;
  onRecall: (record: GenerationRecord) => void;
  onImageToVideo: (request: VideoSourceRequest) => void;
  /** The app-wide "Show hidden" switch (top bar). Hidden items (Settings > Hidden Content) are left out unless it is on. */
  showHidden: boolean;
}

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

export default function LibraryOutput({ queue, onRecall, onImageToVideo, showHidden }: Props) {
  const [tab, setTab] = useState<GenerationKind>('image');
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  // Optional: collapse items with exactly the same prompt into one stack. Off, the list is every item as ever.
  const [grouped, setGrouped] = useState(false);
  // The prompt of the stack that was opened: the list then shows just that prompt's items.
  const [openPrompt, setOpenPrompt] = useState<string | null>(null);
  // The A-or-B comparison of one prompt's items, while it is open.
  const [compare, setCompare] = useState<{ prompt: string; records: GenerationRecord[] } | null>(null);
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
  // What the last Delete moved to the Trash, so its notice can offer to put it back.
  const [trashNotice, setTrashNotice] = useState<{ text: string; ids: number[] } | null>(null);
  // Bumped to make the list load again from the top (after an undo, or a change made elsewhere).
  const [reloadKey, setReloadKey] = useState(0);
  const [infoId, setInfoId] = useState<number | null>(null);
  const [queueCollapsed, setQueueCollapsed] = useState(false);
  // Upscale jobs already merged into the list below (those finished before this page opened are in its load).
  const mergedUpscales = useRef<Set<number> | null>(null);
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

  // Showing stacks right now (grouping is on and none is open): items are stack covers, and the counts are stacks.
  const stacking = grouped && openPrompt === null;
  const listOptions = useMemo<LibraryListOptions>(() => ({ grouped: stacking, prompt: openPrompt }), [stacking, openPrompt]);

  const loadPage = useCallback(
    async (kind: GenerationKind, beforeId: number | null, favorites: boolean, hidden: boolean, ext: string | null, token: number, options: LibraryListOptions) => {
    loadingRef.current = true;
    setLoading(true);
    try {
      const page = await window.kvgenius.listGenerations(kind, PAGE_SIZE, beforeId, favorites, hidden, kind === 'image' ? ext : null, options);
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
    },
    []
  );

  useEffect(() => {
    const token = ++requestToken.current;
    loadingRef.current = false;
    setRecords([]);
    setHasMore(true);
    setSelection(new Map());
    anchorId.current = null;
    setInfoId(null);
    setLightboxIndex(null);
    void loadPage(tab, null, favoritesOnly, showHidden, extension, token, listOptions);
  }, [tab, favoritesOnly, showHidden, extension, reloadKey, listOptions, loadPage]);

  useEffect(() => {
    window.kvgenius
      .countGenerations(favoritesOnly, showHidden, extension, listOptions)
      .then(setCounts)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [favoritesOnly, showHidden, extension, reloadKey, listOptions]);

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
    const last = records[records.length - 1];
    // Stacks are paged by their newest item; everything else by the item itself.
    void loadPage(tab, stacking ? (last.groupNewestId ?? last.id) : last.id, favoritesOnly, showHidden, extension, requestToken.current, listOptions);
  }, [records, tab, stacking, favoritesOnly, showHidden, extension, listOptions, loadPage]);

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

  function handleTabChange(next: GenerationKind) {
    if (next === tab) return;
    // An opened stack belongs to the tab it was opened in.
    setOpenPrompt(null);
    setTab(next);
  }

  function handleToggleGrouped() {
    setOpenPrompt(null);
    setGrouped((v) => !v);
  }

  /** Opens the A-or-B comparison for everything with this prompt (not just the part that is scrolled into view). */
  async function handleCompare(prompt: string) {
    setBusy(true);
    setError(null);
    try {
      const items: GenerationRecord[] = [];
      let before: number | null = null;
      for (;;) {
        const page: GenerationRecord[] = await window.kvgenius.listGenerations(
          tab,
          200,
          before,
          favoritesOnly,
          showHidden,
          tab === 'image' ? extension : null,
          { prompt }
        );
        items.push(...page);
        if (page.length < 200) break;
        before = page[page.length - 1].id;
      }
      if (items.length < 2) {
        setNotice('There need to be at least two items with this prompt to compare them.');
        return;
      }
      setNotice(null);
      setCompare({ prompt, records: items });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
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
      // Items, never stacks (selecting is off while stacks are shown); inside an opened stack, just that prompt's.
      const refs = await window.kvgenius.listGenerationRefs(tab, favoritesOnly, showHidden, tab === 'image' ? extension : null, {
        prompt: openPrompt,
      });
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

  /** A stack of more than one item, as opposed to a lone item that happens to be shown while grouping. */
  const isStack = (record: GenerationRecord) => stacking && (record.groupCount ?? 1) > 1;

  function handleCardClick(record: GenerationRecord, event: React.MouseEvent) {
    if (!selecting) {
      // Clicking a stack opens it; a lone item opens its details, as everywhere.
      if (isStack(record)) setOpenPrompt(record.prompt);
      else setInfoId(record.id);
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
    if (stacking) {
      // Taking an item out of a stack changes its count and maybe its cover, so load the stacks again.
      setReloadKey((k) => k + 1);
      return;
    }
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
      if (stacking) {
        // It has its source's prompt, so it joins a stack: load the stacks again to show that.
        if (fits) setReloadKey((k) => k + 1);
        continue;
      }
      // Inside an opened stack, only what has that prompt belongs.
      if (openPrompt !== null && made.prompt !== openPrompt) continue;
      if (tab === job.kind && !favoritesOnly && fits) {
        setRecords((prev) => (prev.some((r) => r.id === made.id) ? prev : [made, ...prev]));
      }
      if (fits) setCounts((prev) => ({ ...prev, [job.kind]: prev[job.kind] + 1 }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue.jobs, tab, favoritesOnly, showHidden, extension, stacking, openPrompt]);

  /** A GIF was made from a video: it is a new image, so fold it into the list as it is filtered. */
  function handleGifMade(made: GenerationRecord) {
    const fits = matchesExtension(made) && (!made.hidden || showHidden);
    refreshExtensions();
    if (stacking) {
      // It has its source's prompt, so it joins a stack: load the stacks again to show that.
      if (fits) setReloadKey((k) => k + 1);
      return;
    }
    if (openPrompt !== null && made.prompt !== openPrompt) return;
    if (tab === 'image' && !favoritesOnly && fits) {
      setRecords((prev) => [made, ...prev]);
    }
    if (fits) setCounts((prev) => ({ ...prev, image: prev.image + 1 }));
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

  // Delete moves to the Trash with no confirmation: it can be undone right here, or restored later from
  // Library > Trash. Favorites and pinned items go too - the user asked for these ones.
  async function handleDelete(record: GenerationRecord) {
    try {
      const result = await window.kvgenius.trashGenerations([record.id], { includeKept: true });
      if (result.moved === 0) {
        setError('Could not move it to the Trash.');
        return;
      }
      forgetIds([record.id], kindOf(record));
      setTrashNotice({ text: 'Moved to the Trash.', ids: [record.id] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteSelected() {
    const ids = [...selection.keys()];
    if (ids.length === 0) return;
    try {
      const result = await window.kvgenius.trashGenerations(ids, { includeKept: true });
      exitSelectMode();
      if (result.failed > 0) {
        // Some could not be moved, and which ones is not known here: show the list as it now is.
        setReloadKey((k) => k + 1);
        setError(`${result.failed} item${result.failed === 1 ? '' : 's'} could not be moved to the Trash.`);
      } else {
        forgetIds(ids, tab);
      }
      if (result.moved > 0) setTrashNotice({ text: `Moved ${result.moved} item${result.moved === 1 ? '' : 's'} to the Trash.`, ids });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleUndoTrash() {
    if (!trashNotice) return;
    const ids = trashNotice.ids;
    setTrashNotice(null);
    try {
      const result = await window.kvgenius.restoreGenerations(ids);
      setReloadKey((k) => k + 1);
      if (result.failed > 0) setError(`${result.failed} item${result.failed === 1 ? '' : 's'} could not be restored.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  // The undo offer lapses after a while.
  useEffect(() => {
    if (!trashNotice) return;
    const timer = setTimeout(() => setTrashNotice(null), 15000);
    return () => clearTimeout(timer);
  }, [trashNotice]);

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
    const stack = isStack(record);
    return (
      <div
        key={record.id}
        className={`library-card${selected ? ' library-card--selected' : ''}${active ? ' library-card--active' : ''}${stack ? ' library-card--stack' : ''}`}
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
            {stack && (
              <span className="library-card__stack-badge" title={`${record.groupCount} items have this exact prompt - click to open them`}>
                × {record.groupCount}
              </span>
            )}
            {record.hidden && <span className={`library-card__hidden-badge${stack ? ' library-card__hidden-badge--below' : ''}`}>Hidden</span>}
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
        {!selecting && stack && (
          <div className="library-card__actions">
            <button type="button" onClick={() => setInfoId(record.id)} title="Details of the cover">
              ℹ️
            </button>
            <button type="button" onClick={() => void handleCompare(record.prompt)} disabled={busy} title="Compare these two at a time - A or B? - to find the best">
              ⚖️
            </button>
            <button type="button" className="primary" onClick={() => setOpenPrompt(record.prompt)} title="Show the items with this prompt">
              Open {record.groupCount}
            </button>
          </div>
        )}
        {!selecting && !stack && (
          <div className="library-card__actions">
            <button type="button" onClick={() => setInfoId(record.id)} title="Info">
              ℹ️
            </button>
            {!isVideo && (
              <button type="button" onClick={() => handleImageToVideo(record)} title="Create video from image">
                🎬
              </button>
            )}
            {!isVideo && <CopyButton imagePath={record.imagePath} compact title="Copy the image" />}
            <button type="button" onClick={() => handleSaveAs(record)} title="Save As...">
              💾
            </button>
            <button type="button" onClick={() => handleReveal(record)} title="Show in File Manager">
              📂
            </button>
            <button type="button" onClick={() => handleDelete(record)} title="Delete (moves to the Trash)">
              🗑️
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="library-output">
      {/* Queue on the left, details on the right, so both can be open at once. */}
      <LibraryQueue
        queue={queue}
        collapsed={queueCollapsed}
        onToggle={() => setQueueCollapsed((v) => !v)}
        onToggleFavorite={handleToggleFavorite}
        onRerack={handleRecreate}
      />

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
                className={grouped ? 'primary' : undefined}
                onClick={handleToggleGrouped}
                title="Collapse items with exactly the same prompt into one stack (off: every item is shown)"
              >
                {grouped ? '📚 Grouped by prompt' : '📚 Group by prompt'}
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
              <button
                type="button"
                onClick={() => setSelecting(true)}
                disabled={records.length === 0 || stacking}
                title={stacking ? 'Open a stack, or turn off grouping, to select items' : undefined}
              >
                Select Multiple
              </button>
            </>
          )}
        </div>

        {notice && <p className="library-notice">{notice}</p>}
        {trashNotice && (
          <p className="library-notice">
            {trashNotice.text}{' '}
            <button type="button" onClick={handleUndoTrash}>
              ↶ Undo
            </button>
          </p>
        )}

        <div className="tab-strip" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'image'}
            className={`tab-strip__tab${tab === 'image' ? ' active' : ''}`}
            onClick={() => handleTabChange('image')}
          >
            🖼️ Images ({counts.image}{stacking ? ' prompts' : ''})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'video'}
            className={`tab-strip__tab${tab === 'video' ? ' active' : ''}`}
            onClick={() => handleTabChange('video')}
          >
            🎬 Videos ({counts.video}{stacking ? ' prompts' : ''})
          </button>
        </div>

        {openPrompt !== null && (
          <div className="stack-header">
            <button type="button" onClick={() => setOpenPrompt(null)} title="Back to the stacks">
              ← All prompts
            </button>
            <span className="stack-header__prompt" title={openPrompt}>
              {openPrompt}
            </span>
            <span className="stack-header__count">
              {counts[tab]} item{counts[tab] === 1 ? '' : 's'}
            </span>
            <button
              type="button"
              className="primary"
              onClick={() => void handleCompare(openPrompt)}
              disabled={busy || counts[tab] < 2}
              title="Pick the best of these: two at a time, A or B?"
            >
              ⚖️ Compare
            </button>
          </div>
        )}

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
        <LibraryDetails
          record={infoRecord}
          queue={queue}
          onClose={() => setInfoId(null)}
          onExpand={() => setLightboxIndex(records.findIndex((r) => r.id === infoRecord.id))}
          onToggleFavorite={handleToggleFavorite}
          onTogglePinned={handleTogglePinned}
          onToggleHidden={handleToggleHidden}
          onDelete={handleDelete}
          onRerack={handleRecreate}
          onImageToVideo={handleImageToVideo}
          onSaveAs={handleSaveAs}
          onReveal={handleReveal}
          onUpscaleQueued={() => setQueueCollapsed(false)}
          onGifMade={handleGifMade}
          onError={setError}
          onNotice={setNotice}
        />
      )}

      {compare && (
        <CompareOverlay
          prompt={compare.prompt}
          records={compare.records}
          onClose={(changed) => {
            setCompare(null);
            // A favorite, pin or trash made in there: show the list as it now is.
            if (changed) setReloadKey((k) => k + 1);
          }}
        />
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
