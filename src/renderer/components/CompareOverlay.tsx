import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FAMILY_KIND, GenerationRecord } from '../../shared/types';
import { Pick, answer, answered, currentPair, isDone, questionsLeft, startTournament, undo, winner } from '../../shared/tournament';
import CopyButton from './CopyButton';
import GeneratedVideo from './GeneratedVideo';

interface Props {
  /** The prompt every item shares. */
  prompt: string;
  /** The items to choose between (same prompt, same kind). */
  records: GenerationRecord[];
  /** `changed` is true if anything was favorited, pinned or trashed, so the page behind should reload. */
  onClose: (changed: boolean) => void;
}

const isVideo = (record: GenerationRecord) => FAMILY_KIND[record.modelFamily] === 'video';

/**
 * Picks the best of a set the eye-doctor way: A or B? The pick stays and is shown against the next,
 * until what is left is the best (see shared/tournament.ts). Keys: ← picks A, → picks B, ↓ neither,
 * Backspace undoes, Esc leaves. At the end the best can be favorited or pinned, and the rest moved to the
 * Trash (never a favorite or a pinned item, and it can be undone).
 */
export default function CompareOverlay({ prompt, records, onClose }: Props) {
  const ids = useMemo(() => records.map((r) => r.id), [records]);
  const [state, setState] = useState(() => startTournament(ids));
  // Favoriting moves a file and changes its path, so what the overlay shows is the records plus what it changed.
  const [patches, setPatches] = useState<Record<number, Partial<GenerationRecord>>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trashed, setTrashed] = useState<{ ids: number[]; text: string } | null>(null);
  const changed = useRef(false);

  const get = (id: number): GenerationRecord => {
    const record = records.find((r) => r.id === id) as GenerationRecord;
    return { ...record, ...patches[id] };
  };
  const patch = (id: number, changes: Partial<GenerationRecord>) => setPatches((prev) => ({ ...prev, [id]: { ...prev[id], ...changes } }));
  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));

  const pair = currentPair(state);
  const best = winner(state);
  const done = isDone(state);

  function choose(pick: Pick) {
    if (busy || done) return;
    setState((s) => answer(s, pick));
  }

  function handleUndo() {
    setState((s) => undo(s));
  }

  // Keys. Left/Right/Down choose, Backspace takes the last answer back, Escape leaves.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose(changed.current);
      else if (busy) return;
      else if (e.key === 'ArrowLeft') choose('a');
      else if (e.key === 'ArrowRight') choose('b');
      else if (e.key === 'ArrowDown' || e.key.toLowerCase() === 'n') choose('neither');
      else if (e.key === 'Backspace' || e.key.toLowerCase() === 'z') handleUndo();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, done, onClose]);

  // Load the picture that comes next while this question is being decided, so it appears at once.
  const upcoming = state.champion !== null ? state.challengers[1] : state.challengers[2];
  useEffect(() => {
    if (upcoming === undefined) return;
    const record = records.find((r) => r.id === upcoming);
    if (record && !isVideo(record)) new Image().src = window.kvgenius.imageUrlFor(record.imagePath);
  }, [upcoming, records]);

  function renderMedia(record: GenerationRecord, big: boolean) {
    const url = window.kvgenius.imageUrlFor(record.imagePath);
    if (isVideo(record)) {
      // A looping, muted preview while choosing; the best one gets real controls.
      return (
        <GeneratedVideo
          src={url}
          filePath={record.imagePath}
          thumbnail={!big}
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
      );
    }
    return <img src={url} alt={record.prompt} draggable={false} />;
  }

  function renderPane(id: number, side: 'a' | 'b') {
    const record = get(id);
    return (
      <div
        key={id}
        className="compare__pane"
        onClick={() => choose(side)}
        title={`This one is better (${side === 'a' ? '←' : '→'})`}
      >
        <span className="compare__label">{side.toUpperCase()}</span>
        {renderMedia(record, false)}
        {!isVideo(record) && <CopyButton compact imagePath={record.imagePath} className="compare__copy" title="Copy this image" />}
        <span className="compare__meta">
          seed {record.seed} · {record.width}×{record.height}
        </span>
      </div>
    );
  }

  async function handleToggleFavorite(id: number) {
    const record = get(id);
    setError(null);
    try {
      const { imagePath } = await window.kvgenius.setGenerationFavorite(id, !record.favorite);
      patch(id, { favorite: !record.favorite, imagePath });
      changed.current = true;
    } catch (err) {
      fail(err);
    }
  }

  async function handleTogglePinned(id: number) {
    const record = get(id);
    setError(null);
    try {
      await window.kvgenius.setGenerationPinned(id, !record.pinned);
      patch(id, { pinned: !record.pinned });
      changed.current = true;
    } catch (err) {
      fail(err);
    }
  }

  // Everything but the best (or, if nothing was good enough, everything) - never a favorite or a pinned item.
  const rest = best === null ? ids : ids.filter((id) => id !== best);
  const protectedCount = rest.filter((id) => get(id).favorite || get(id).pinned).length;

  async function handleTrashRest() {
    if (rest.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.trashGenerations(rest);
      changed.current = true;
      setTrashed({
        ids: rest,
        text:
          `Moved ${result.moved} to the Trash.` + (result.skipped > 0 ? ` ${result.skipped} stayed (favorites or pinned).` : '') +
          (result.failed > 0 ? ` ${result.failed} could not be moved.` : ''),
      });
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleUndoTrash() {
    if (!trashed) return;
    const { ids: movedIds } = trashed;
    setTrashed(null);
    try {
      await window.kvgenius.restoreGenerations(movedIds);
    } catch (err) {
      fail(err);
    }
  }

  const askedSoFar = answered(state);
  const left = questionsLeft(state);
  const bestRecord = best === null ? null : get(best);

  return createPortal(
    <div className="compare" role="dialog" aria-modal="true" aria-label="Compare">
      <div className="compare__bar">
        <strong className="compare__title">Which is better?</strong>
        <span className="compare__prompt" title={prompt}>
          {prompt}
        </span>
        <span className="compare__progress">
          {done ? `Done - ${state.total} compared` : `Question ${askedSoFar + 1} of up to ${askedSoFar + left}`}
        </span>
        <button type="button" onClick={() => onClose(changed.current)} title="Leave (Esc)">
          ✕ Close
        </button>
      </div>

      {error && <p style={{ color: 'var(--color-accent-red)', margin: '0 16px' }}>{error}</p>}

      {!done && pair && (
        <>
          <div className="compare__panes">
            {renderPane(pair[0], 'a')}
            {renderPane(pair[1], 'b')}
          </div>
          <div className="compare__actions">
            <button type="button" className="primary" onClick={() => choose('a')}>
              ← A is better
            </button>
            <button type="button" onClick={() => choose('neither')} title="Drop both (↓)">
              Neither
            </button>
            <button type="button" className="primary" onClick={() => choose('b')}>
              B is better →
            </button>
            <button type="button" onClick={handleUndo} disabled={askedSoFar === 0} title="Take back the last answer (Backspace)">
              ↶ Undo
            </button>
          </div>
        </>
      )}

      {done && (
        <div className="compare__result">
          {bestRecord ? (
            <>
              <h3 style={{ margin: 0 }}>The best of {state.total}</h3>
              <div className="compare__winner">
                {renderMedia(bestRecord, true)}
                {!isVideo(bestRecord) && <CopyButton compact imagePath={bestRecord.imagePath} className="compare__copy" title="Copy this image" />}
              </div>
            </>
          ) : (
            <h3 style={{ margin: '24px 0' }}>You passed on all {state.total}.</h3>
          )}

          <div className="compare__actions">
            {bestRecord && (
              <>
                <button type="button" onClick={() => handleToggleFavorite(bestRecord.id)}>
                  {bestRecord.favorite ? '★ Favorited' : '☆ Favorite it'}
                </button>
                <button type="button" onClick={() => handleTogglePinned(bestRecord.id)}>
                  {bestRecord.pinned ? '📌 Pinned' : '📌 Pin it'}
                </button>
              </>
            )}
            <button type="button" onClick={handleTrashRest} disabled={busy || rest.length === 0 || trashed !== null}>
              {bestRecord ? `Move the other ${rest.length} to the Trash` : `Move all ${rest.length} to the Trash`}
              {protectedCount > 0 ? ` (${protectedCount} favorite/pinned stay)` : ''}
            </button>
            <button type="button" onClick={() => setState(startTournament(ids))} disabled={busy || trashed !== null}>
              Compare again
            </button>
            <button type="button" className="primary" onClick={() => onClose(changed.current)}>
              Done
            </button>
          </div>
          {trashed && (
            <p className="library-notice" style={{ margin: 0 }}>
              {trashed.text}{' '}
              <button type="button" onClick={handleUndoTrash}>
                ↶ Undo
              </button>
            </p>
          )}
          <button type="button" onClick={handleUndo} disabled={askedSoFar === 0 || trashed !== null} title="Take back the last answer (Backspace)">
            ↶ Undo the last answer
          </button>
        </div>
      )}
    </div>,
    document.body
  );
}
