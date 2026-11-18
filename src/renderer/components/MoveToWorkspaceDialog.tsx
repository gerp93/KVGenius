import { useEffect, useState } from 'react';
import type { MoveResult, WorkspaceSummary } from '../../shared/workspaces';
import './MoveToWorkspaceDialog.css';

type Props = {
  /** The items to move (Library ids). */
  ids: number[];
  /** The workspace they are in now, which is not offered. */
  currentWorkspaceId: number;
  onClose: () => void;
  /** Called after the move, with what it did and where the items went. */
  onMoved: (result: MoveResult, target: WorkspaceSummary) => void;
};

/**
 * "Move to workspace": picks the workspace the chosen items go to. Their files move into that workspace's folders, so this is a
 * real move (the items are no longer in this workspace afterwards), not a tag. Hidden workspaces are offered too - that is how
 * items are put out of sight.
 */
export default function MoveToWorkspaceDialog({ ids, currentWorkspaceId, onClose, onMoved }: Props) {
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[] | null>(null);
  const [targetId, setTargetId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    window.kvgenius
      .listWorkspaces(true)
      .then((list) => setWorkspaces(list.filter((w) => w.id !== currentWorkspaceId)))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [currentWorkspaceId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  async function move() {
    const target = workspaces?.find((w) => w.id === targetId);
    if (!target) return;
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.moveGenerations(ids, target.id);
      onMoved(result, target);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  const what = ids.length === 1 ? 'this item' : `these ${ids.length} items`;
  return (
    <div className="move-dialog__backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <div className="move-dialog" role="dialog" aria-modal="true" aria-label="Move to workspace">
        <h3>Move to workspace</h3>
        <p className="move-dialog__lead">
          Move {what} to another workspace. Their files move into that workspace's folders, and they are no longer in this one.
        </p>
        {workspaces === null && !error && <p className="move-dialog__lead">Loading...</p>}
        {workspaces?.length === 0 && <p className="move-dialog__lead">There is no other workspace yet. Make one on the workspace grid (top bar &gt; All workspaces).</p>}
        <div className="move-dialog__list" role="radiogroup" aria-label="Workspace">
          {workspaces?.map((workspace) => (
            <label key={workspace.id} className={`move-dialog__option${targetId === workspace.id ? ' move-dialog__option--on' : ''}`}>
              <input type="radio" name="move-target" checked={targetId === workspace.id} onChange={() => setTargetId(workspace.id)} disabled={busy} />
              <span>
                {workspace.name}
                {workspace.hidden ? ' (hidden)' : ''}
                <span className="move-dialog__count"> - {workspace.itemCount === 1 ? '1 item' : `${workspace.itemCount} items`}</span>
              </span>
            </label>
          ))}
        </div>
        {error && <p className="move-dialog__error">{error}</p>}
        <div className="move-dialog__buttons">
          <button type="button" className="primary" onClick={move} disabled={busy || targetId === null}>
            {busy ? 'Moving...' : 'Move'}
          </button>
          <button type="button" onClick={onClose} disabled={busy}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
