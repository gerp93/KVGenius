import { FormEvent, useCallback, useEffect, useState } from 'react';
import { THEME_NAMES, themeDisplayName } from '../../shared/themes';
import type { WorkspaceSummary } from '../../shared/workspaces';
import './WorkspaceGrid.css';

type Props = {
  /** Opens a workspace: the app then works inside it until it is switched or left. */
  onOpen: (id: number) => void;
};

function itemsLabel(count: number): string {
  return count === 1 ? '1 item' : `${count} items`;
}

/** The edit panel on a workspace's card: its name, its own theme, whether it is hidden, and (when empty) deleting it. */
function EditPanel({
  workspace,
  others,
  onDone,
  onChanged,
  onMerged,
}: {
  workspace: WorkspaceSummary;
  /** The other workspaces on the grid, which this one can be merged into. */
  others: WorkspaceSummary[];
  onDone: () => void;
  onChanged: () => void;
  /** Tells the grid what a merge did (this panel is gone once the workspace is). */
  onMerged: (message: string) => void;
}) {
  const [name, setName] = useState(workspace.name);
  const [theme, setTheme] = useState(workspace.theme ?? '');
  const [hidden, setHidden] = useState(workspace.hidden);
  const [mergeInto, setMergeInto] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await window.kvgenius.updateWorkspace(workspace.id, { name, theme: theme === '' ? null : theme, hidden });
      onChanged();
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function merge() {
    const into = others.find((w) => String(w.id) === mergeInto);
    if (!into) return;
    const items = workspace.itemCount === 1 ? '1 item' : `${workspace.itemCount} items`;
    if (
      !window.confirm(
        `Merge "${workspace.name}" (${items}) into "${into.name}"?

Everything in "${workspace.name}" - its items, their files, its prompts and its own styles and elements - moves into "${into.name}", ` +
          `and "${workspace.name}" is deleted. It cannot be undone as a whole; items can be moved back one at a time.`
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.mergeWorkspaces(workspace.id, into.id);
      const renamed = result.renamedStyles > 0 ? ` ${result.renamedStyles} style${result.renamedStyles === 1 ? '' : 's'} got "(from ${workspace.name})" added to the name because "${into.name}" already had one with that name.` : '';
      if (result.deleted) {
        onMerged(`Merged "${workspace.name}" into "${into.name}": ${result.moved} item${result.moved === 1 ? '' : 's'} moved.${renamed}`);
      } else {
        setError(`${result.moved} moved, but ${result.failed} could not be moved (their files may be in use), so "${workspace.name}" was kept with what is left. Close whatever is using them and merge again.`);
      }
      onChanged();
      if (result.deleted) onDone();
      else setBusy(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  async function remove() {
    if (!window.confirm(`Delete the workspace "${workspace.name}"? This cannot be undone.`)) return;
    setBusy(true);
    setError(null);
    try {
      await window.kvgenius.deleteWorkspace(workspace.id);
      onChanged();
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <form className="workspace-edit" onSubmit={save}>
      <label className="field-label" htmlFor={`ws-name-${workspace.id}`}>
        Name
      </label>
      <input id={`ws-name-${workspace.id}`} value={name} onChange={(e) => setName(e.target.value)} maxLength={60} autoFocus />
      <label className="field-label" htmlFor={`ws-theme-${workspace.id}`}>
        Theme
      </label>
      <select id={`ws-theme-${workspace.id}`} value={theme} onChange={(e) => setTheme(e.target.value)}>
        <option value="">Same as the app</option>
        {THEME_NAMES.map((id) => (
          <option key={id} value={id}>
            {themeDisplayName(id)}
          </option>
        ))}
      </select>
      <label className="workspace-edit__check">
        <input type="checkbox" checked={hidden} onChange={(e) => setHidden(e.target.checked)} />
        <span>Hidden - left off this grid and the switcher unless "Show hidden" is on</span>
      </label>
      {error && <p className="workspace-edit__error">{error}</p>}
      <div className="workspace-edit__buttons">
        <button type="submit" className="primary" disabled={busy}>
          Save
        </button>
        <button type="button" onClick={onDone} disabled={busy}>
          Cancel
        </button>
        <button type="button" className="workspace-edit__delete" onClick={remove} disabled={busy} title="Only a workspace with nothing in it can be deleted">
          Delete
        </button>
      </div>
      {others.length > 0 && (
        <div className="workspace-edit__merge">
          <label className="field-label" htmlFor={`ws-merge-${workspace.id}`}>
            Merge into another workspace
          </label>
          <select id={`ws-merge-${workspace.id}`} value={mergeInto} onChange={(e) => setMergeInto(e.target.value)} disabled={busy}>
            <option value="">Choose one...</option>
            {others.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
                {w.hidden ? ' (hidden)' : ''}
              </option>
            ))}
          </select>
          <button type="button" onClick={merge} disabled={busy || mergeInto === ''}>
            Merge
          </button>
        </div>
      )}
    </form>
  );
}

/**
 * The first screen: every workspace as a card. Opening one puts the whole app inside it (see shared/workspaces.ts); the
 * switcher in the top bar changes workspace from anywhere, and brings you back here. Hidden workspaces are only listed
 * while "Show hidden" is on, which is not remembered - they start out of sight every time.
 */
export default function WorkspaceGrid({ onOpen }: Props) {
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[] | null>(null);
  const [showHidden, setShowHidden] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const reload = useCallback(() => {
    window.kvgenius
      .listWorkspaces(showHidden)
      .then(setWorkspaces)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [showHidden]);

  useEffect(() => {
    reload();
  }, [reload]);

  async function create(event: FormEvent) {
    event.preventDefault();
    setCreating(true);
    setError(null);
    try {
      await window.kvgenius.createWorkspace(newName);
      setNewName('');
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="workspace-grid-page">
      <header className="workspace-grid-page__head">
        <div>
          <h1>KVGenius</h1>
          <p className="workspace-grid-page__lead">Choose a workspace. Each one is its own library of images, videos and prompts.</p>
        </div>
        <label className="workspace-grid-page__toggle">
          <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} />
          <span>Show hidden</span>
        </label>
      </header>

      {notice && <p className="workspace-grid-page__lead">{notice}</p>}
      {error && <p className="workspace-edit__error">{error}</p>}
      {workspaces === null && !error && <p className="workspace-grid-page__lead">Loading...</p>}

      <div className="workspace-grid">
        {workspaces?.map((workspace) => (
          <div key={workspace.id} className={`workspace-card${workspace.hidden ? ' workspace-card--hidden' : ''}`}>
            {editingId === workspace.id ? (
              <EditPanel
                workspace={workspace}
                others={workspaces.filter((w) => w.id !== workspace.id)}
                onDone={() => setEditingId(null)}
                onChanged={reload}
                onMerged={setNotice}
              />
            ) : (
              <>
                <button type="button" className="workspace-card__open" onClick={() => onOpen(workspace.id)} title={`Open ${workspace.name}`}>
                  <span className="workspace-card__cover">
                    {workspace.coverPath ? <img src={window.kvgenius.imageUrlFor(workspace.coverPath)} alt="" loading="lazy" /> : <span className="workspace-card__empty">Empty</span>}
                  </span>
                  <span className="workspace-card__name">{workspace.name}</span>
                  <span className="workspace-card__meta">
                    {itemsLabel(workspace.itemCount)}
                    {workspace.theme ? ` · ${themeDisplayName(workspace.theme)} theme` : ''}
                    {workspace.hidden ? ' · Hidden' : ''}
                  </span>
                </button>
                <button type="button" className="workspace-card__edit" onClick={() => setEditingId(workspace.id)}>
                  Edit
                </button>
              </>
            )}
          </div>
        ))}

        <form className="workspace-card workspace-card--new" onSubmit={create}>
          <label className="field-label" htmlFor="new-workspace-name">
            New workspace
          </label>
          <input id="new-workspace-name" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Name" maxLength={60} />
          <button type="submit" className="primary" disabled={creating || newName.trim() === ''}>
            Create
          </button>
        </form>
      </div>
    </div>
  );
}
