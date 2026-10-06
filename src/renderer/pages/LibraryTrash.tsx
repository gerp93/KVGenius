import { useCallback, useEffect, useRef, useState } from 'react';
import { FAMILY_KIND, GenerationKind, GenerationRecord } from '../../shared/types';
import { TrashStats } from '../../shared/cleanup';
import GalleryLightbox from '../components/GalleryLightbox';
import GeneratedVideo from '../components/GeneratedVideo';
import { formatBytes } from '../utils/format';
import { justifyRows } from '../utils/justifiedRows';

const PAGE_SIZE = 100;
const TARGET_ROW_HEIGHT = 200;
const GRID_GAP = 12;

function kindOf(record: GenerationRecord): GenerationKind {
  return FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
}

function plural(n: number): string {
  return `${n} item${n === 1 ? '' : 's'}`;
}

/** Library > Trash: what was moved out of the Library, with Restore and Delete forever. Deleting from
 * here is the only thing in the app that removes a trashed file, and it asks first. */
export default function LibraryTrash() {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [stats, setStats] = useState<TrashStats>({ count: 0, bytes: 0 });
  const [loaded, setLoaded] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [gridWidth, setGridWidth] = useState(0);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [selection, setSelection] = useState<Set<number>>(new Set());
  // The card a Shift-click range starts from: the last one clicked in select mode.
  const anchorId = useRef<number | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));

  const load = useCallback(async (beforeId: number | null) => {
    try {
      const page = await window.kvgenius.listTrashed(PAGE_SIZE, beforeId);
      setRecords((prev) => (beforeId === null ? page : [...prev, ...page]));
      setHasMore(page.length === PAGE_SIZE);
      setLoaded(true);
    } catch (err) {
      fail(err);
    }
  }, []);

  const refreshStats = useCallback(() => {
    window.kvgenius.getTrashStats().then(setStats).catch(fail);
  }, []);

  useEffect(() => {
    void load(null);
    refreshStats();
  }, [load, refreshStats]);

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

  function forget(ids: number[]) {
    const gone = new Set(ids);
    setRecords((prev) => prev.filter((r) => !gone.has(r.id)));
    setSelection((prev) => new Set([...prev].filter((id) => !gone.has(id))));
    setLightboxIndex(null);
    refreshStats();
  }

  function exitSelectMode() {
    setSelecting(false);
    setSelection(new Set());
    anchorId.current = null;
  }

  function toggleSelected(record: GenerationRecord) {
    setSelection((prev) => {
      const next = new Set(prev);
      if (next.has(record.id)) next.delete(record.id);
      else next.add(record.id);
      return next;
    });
  }

  /** Click toggles a card; Shift-click selects the whole run from the last card clicked (nothing is deselected). */
  function handleCardClick(record: GenerationRecord, index: number, event: React.MouseEvent) {
    if (!selecting) {
      setLightboxIndex(index);
      return;
    }
    const from = records.findIndex((r) => r.id === anchorId.current);
    if (event.shiftKey && from >= 0) {
      const [lo, hi] = from < index ? [from, index] : [index, from];
      setSelection((prev) => new Set([...prev, ...records.slice(lo, hi + 1).map((r) => r.id)]));
    } else {
      toggleSelected(record);
      anchorId.current = record.id;
    }
  }

  /** Every id in the Trash - not just the loaded page. */
  async function allTrashedIds(): Promise<number[]> {
    const ids: number[] = [];
    let before: number | null = null;
    for (;;) {
      const page: GenerationRecord[] = await window.kvgenius.listTrashed(200, before);
      ids.push(...page.map((r) => r.id));
      if (page.length < 200) break;
      before = page[page.length - 1].id;
    }
    return ids;
  }

  async function handleSelectAll() {
    setBusy(true);
    try {
      setSelection(new Set(await allTrashedIds()));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleRestoreSelected() {
    const ids = [...selection];
    if (ids.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.restoreGenerations(ids);
      setNotice(`Restored ${plural(result.restored)}.` + (result.failed > 0 ? ` ${plural(result.failed)} could not be restored.` : ''));
      exitSelectMode();
      await load(null);
      refreshStats();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleRecycleSelected() {
    const ids = [...selection];
    if (ids.length === 0) return;
    if (
      !window.confirm(
        `Send ${plural(ids.length)} to the Recycle Bin?\n\nThey can no longer be restored into the app, but you can still get the files back from the Recycle Bin.`
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.deleteTrashed(ids);
      setNotice(
        `Sent ${plural(result.deleted)} to the Recycle Bin.` + (result.failed > 0 ? ` ${plural(result.failed)} could not be sent and stay in the Trash.` : '')
      );
      exitSelectMode();
      await load(null);
      refreshStats();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleRestore(record: GenerationRecord) {
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.restoreGenerations([record.id]);
      if (result.restored > 0) {
        forget([record.id]);
        setNotice('Restored - it is back in Library > Output.');
      } else {
        setError('Could not restore it: its file is no longer in the Trash.');
      }
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  /** Step two for one item: its file goes to the Recycle Bin and it leaves the app for good. */
  async function handleDelete(record: GenerationRecord) {
    if (
      !window.confirm(
        'Send this to the Recycle Bin?\n\nIt can no longer be restored into the app, but you can still get the file back from the Recycle Bin.'
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.deleteTrashed([record.id]);
      if (result.deleted > 0) {
        forget([record.id]);
        setNotice('Sent to the Recycle Bin.');
      } else {
        setError('The Recycle Bin would not take that file, so it stays in the Trash.');
      }
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleRestoreAll() {
    if (records.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      // Everything, not just the loaded page.
      const ids = await allTrashedIds();
      const result = await window.kvgenius.restoreGenerations(ids);
      setNotice(`Restored ${plural(result.restored)}.` + (result.failed > 0 ? ` ${plural(result.failed)} could not be restored.` : ''));
      await load(null);
      refreshStats();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleEmpty() {
    if (stats.count === 0) return;
    if (
      !window.confirm(
        `Send ${plural(stats.count)} (${formatBytes(stats.bytes)}) from the Trash to the Recycle Bin?\n\nThey can no longer be restored into the app, ` +
          `but you can still get the files back from the Recycle Bin.`
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.emptyTrash();
      await load(null);
      setNotice(
        `Sent ${plural(result.deleted)} to the Recycle Bin.` + (result.failed > 0 ? ` ${plural(result.failed)} could not be sent and stay in the Trash.` : '')
      );
      refreshStats();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="library-output">
      <div className="library-output__main">
        {/* Stays at the top of the list while scrolling. */}
        <div className="library-sticky">
          {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

          <div className="prompt-library__toolbar">
            <strong>Trash</strong>
            <span className="prompt-library__count">
              {plural(stats.count)}
              {stats.count > 0 ? ` - ${formatBytes(stats.bytes)}` : ''}
            </span>
            <span style={{ flex: 1 }} />
            {selecting ? (
              <>
                <span className="library-toolbar__hint">Click to select, Shift-click for a range</span>
                <button type="button" onClick={handleSelectAll} disabled={busy || stats.count === 0 || selection.size === stats.count}>
                  Select All ({stats.count})
                </button>
                <button type="button" onClick={() => setSelection(new Set())} disabled={busy || selection.size === 0}>
                  Clear Selection
                </button>
                <button type="button" className="primary" onClick={handleRestoreSelected} disabled={busy || selection.size === 0}>
                  Restore Selected ({selection.size})
                </button>
                <button type="button" onClick={handleRecycleSelected} disabled={busy || selection.size === 0}>
                  ♻️ Recycle Selected ({selection.size})
                </button>
                <button type="button" onClick={exitSelectMode}>
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button type="button" onClick={() => setSelecting(true)} disabled={busy || records.length === 0}>
                  Select Multiple
                </button>
                <button type="button" onClick={handleRestoreAll} disabled={busy || stats.count === 0}>
                  Restore All
                </button>
                <button type="button" onClick={handleEmpty} disabled={busy || stats.count === 0}>
                  Empty Trash (to Recycle Bin)
                </button>
              </>
            )}
          </div>

          {notice && <p className="library-notice">{notice}</p>}
        </div>

        {loaded && records.length === 0 && (
          <p style={{ color: 'var(--color-text-muted)' }}>
            The Trash is empty. Anything you delete, or clean up from Settings, waits here until you restore it or
            send it to the Recycle Bin.
          </p>
        )}

        {/* user-select is off in select mode so Shift-click picks a range instead of highlighting text */}
        <div className={`library-rows${selecting ? ' library-rows--selecting' : ''}`} ref={gridRef}>
          {rows.map((row) => (
            <div key={records[row.items[0].index].id} className="library-row">
              {row.items.map(({ index, width }) => {
                const record = records[index];
                const url = window.kvgenius.imageUrlFor(record.imagePath);
                const selected = selecting && selection.has(record.id);
                return (
                  <div
                    key={record.id}
                    className={`library-card prompt-tile${selected ? ' library-card--selected' : ''}`}
                    style={{ width }}
                  >
                    {selecting && (
                      <span className="library-card__select">
                        <input type="checkbox" checked={selected} readOnly tabIndex={-1} />
                      </span>
                    )}
                    <div
                      className="library-card__media"
                      style={{ height: row.height, cursor: 'pointer' }}
                      onClick={(e) => handleCardClick(record, index, e)}
                    >
                      {kindOf(record) === 'video' ? (
                        <>
                          <GeneratedVideo src={url} filePath={record.imagePath} thumbnail />
                          <span className="library-card__play-badge">▶</span>
                        </>
                      ) : (
                        <img src={url} alt={record.prompt} loading="lazy" decoding="async" />
                      )}
                      <div className="prompt-tile__prompt">{record.prompt}</div>
                    </div>
                    {!selecting && (
                      <div className="library-card__actions library-card__actions--labeled">
                        <button type="button" className="primary" onClick={() => handleRestore(record)} disabled={busy} title="Put it back in the Library">
                          Restore
                        </button>
                        <button type="button" onClick={() => handleDelete(record)} disabled={busy} title="Send to the Recycle Bin (it can no longer be restored into the app)">
                          ♻️ Recycle
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {hasMore && (
          <div style={{ padding: 12, textAlign: 'center' }}>
            <button type="button" onClick={() => void load(records[records.length - 1].id)}>
              Load more
            </button>
          </div>
        )}
      </div>

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
