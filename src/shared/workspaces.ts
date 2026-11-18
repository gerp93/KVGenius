import { CleanupSettings } from './cleanup';

/**
 * A workspace is a whole, separate library: its own images, videos, pinned prompts, kept sources,
 * trash and queue history, its own elements, and optionally its own theme and cleanup rules. The app
 * opens on a grid of them, and everything below that grid works inside one workspace at a time.
 *
 * Styles can be shared across workspaces (a style with no workspace is global); everything else made
 * or kept belongs to exactly one workspace. A hidden workspace is left off the grid and the switcher
 * until "Show hidden" is on - hiding is a property of a whole workspace, never of single items.
 */
export interface Workspace {
  id: number;
  name: string;
  /** This workspace's theme, or null to follow the app's theme (Settings). */
  theme: string | null;
  hidden: boolean;
  /** Retention rules for this workspace's Library and Trash (Settings > Library Cleanup). */
  cleanup: CleanupSettings;
  createdAt: string;
}

/** A workspace as the grid shows it. */
export interface WorkspaceSummary extends Workspace {
  /** Items in the Library (not the Trash). */
  itemCount: number;
  /** The newest picture, to put on the card; null while there is none. */
  coverPath: string | null;
}

export interface WorkspaceUpdate {
  name?: string;
  /** A theme id, or null to follow the app's theme. */
  theme?: string | null;
  hidden?: boolean;
}

export const DEFAULT_WORKSPACE_NAME = 'Default';
/** Where the Styles page keeps its example pictures: never listed, so they never show up in a Library. */
export const STYLE_EXAMPLES_WORKSPACE_NAME = 'Style examples';
export const HIDDEN_WORKSPACE_NAME = 'Hidden';
export const MAX_WORKSPACE_NAME_LENGTH = 60;

/** A workspace name cleaned up (spaces trimmed and collapsed), or the reason it is not allowed. */
export function validateWorkspaceName(raw: unknown): { ok: true; name: string } | { ok: false; message: string } {
  if (typeof raw !== 'string') return { ok: false, message: 'A workspace needs a name.' };
  const name = raw.trim().replace(/\s+/g, ' ');
  if (name === '') return { ok: false, message: 'A workspace needs a name.' };
  if (name.length > MAX_WORKSPACE_NAME_LENGTH) return { ok: false, message: `A workspace name can be at most ${MAX_WORKSPACE_NAME_LENGTH} characters.` };
  return { ok: true, name };
}

/** Which workspace a style or element belongs to, as the Styles page words it. */
export function styleScopeLabel(workspaceId: number | null): string {
  return workspaceId === null ? 'All workspaces' : 'This workspace';
}
