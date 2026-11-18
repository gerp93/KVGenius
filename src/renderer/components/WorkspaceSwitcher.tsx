import { useEffect, useState } from 'react';
import type { Workspace } from '../../shared/workspaces';

type Props = {
  /** The workspace the app is in. */
  current: Workspace;
  /** Moves the app into another workspace. */
  onSwitch: (id: number) => void;
  /** Goes back to the grid of all workspaces (where they are made, renamed and themed). */
  onShowAll: () => void;
};

const SHOW_ALL = 'all';

/** The top bar's workspace switcher: the current workspace, with the others to jump to. Hidden workspaces are not offered here
 * (the grid's "Show hidden" is the way to one), except the one you are already in. */
export default function WorkspaceSwitcher({ current, onSwitch, onShowAll }: Props) {
  const [others, setOthers] = useState<Workspace[]>([]);

  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listWorkspaces(false)
      .then((list) => {
        if (!cancelled) setOthers(list);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [current.id]);

  const options = others.some((w) => w.id === current.id) ? others : [current, ...others];

  return (
    <label className="top-bar__workspace" title="Switch workspace">
      <span aria-hidden="true">🗂️</span>
      <select
        aria-label="Workspace"
        value={current.id}
        onChange={(e) => {
          if (e.target.value === SHOW_ALL) onShowAll();
          else onSwitch(Number(e.target.value));
        }}
      >
        {options.map((workspace) => (
          <option key={workspace.id} value={workspace.id}>
            {workspace.name}
          </option>
        ))}
        <option value={SHOW_ALL}>All workspaces...</option>
      </select>
    </label>
  );
}
