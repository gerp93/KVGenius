import { DatabaseSync } from 'node:sqlite';
import { CleanupSettings, autoEmptyDue, autoTrashDue } from '../shared/cleanup';
import { Workspace } from '../shared/workspaces';
import { Recycle, emptyTrash, runCleanup } from './trash';

/** What the scheduled cleanup needs from the app, passed in so it can run (and be tested) on its own. */
export interface CleanupDeps {
  getDb: () => DatabaseSync | null;
  /** Every workspace with its own retention rules (hidden ones included). */
  listWorkspaces: (db: DatabaseSync) => Workspace[];
  /** Saves a workspace's rules, which is how a run is remembered. */
  saveSettings: (db: DatabaseSync, workspaceId: number, settings: CleanupSettings) => void;
  /** Reads a workspace's rules again (they can change while files are being sent). */
  getSettings: (db: DatabaseSync, workspaceId: number) => CleanupSettings | null;
  trashDir: () => string;
  sourcesDir: () => string;
  /** Sends a file to the operating system's Recycle Bin. */
  recycle: Recycle;
}

function plural(n: number): string {
  return `${n} item${n === 1 ? '' : 's'}`;
}

/** What the two schedules did for one workspace on this check: a sentence each, or null if that one did not run. */
export interface AutoCleanupResult {
  workspaceId: number;
  trashed: string | null;
  emptied: string | null;
}

/**
 * Runs, for each workspace, whichever automatic step its own settings turned on and that is due (each only if the user turned it on,
 * and not more than once a day):
 *  - the Trash is emptied: items that have been in it long enough go to the Recycle Bin;
 *  - old items nobody kept are moved to the Trash.
 * The two are separate options and are checked separately, and one workspace's rules never touch another workspace's items.
 */
export async function runAutoCleanupIfDue(deps: CleanupDeps, now: Date = new Date()): Promise<AutoCleanupResult[]> {
  const results: AutoCleanupResult[] = [];
  const db = deps.getDb();
  if (!db) return results;

  for (const workspace of deps.listWorkspaces(db)) {
    const result: AutoCleanupResult = { workspaceId: workspace.id, trashed: null, emptied: null };
    const start = workspace.cleanup;
    if (autoEmptyDue(start, now)) {
      const emptied = await emptyTrash(db, workspace.id, deps.sourcesDir(), deps.recycle, start.trashRetentionDays, now);
      result.emptied =
        `Sent ${plural(emptied.deleted)} to the Recycle Bin.` + (emptied.failed > 0 ? ` ${plural(emptied.failed)} could not be sent.` : '');
      // Settings can change while the files are being sent, so re-read before saving.
      const latest = deps.getSettings(db, workspace.id);
      if (latest) deps.saveSettings(db, workspace.id, { ...latest, lastAutoEmptyRun: now.toISOString(), lastAutoEmptySummary: result.emptied });
    }

    const current = deps.getSettings(db, workspace.id);
    if (current && autoTrashDue(current, now)) {
      const moved = runCleanup(db, workspace.id, current.olderThanDays, deps.trashDir(), now);
      result.trashed = `Moved ${plural(moved.moved)} to the Trash.` + (moved.failed > 0 ? ` ${plural(moved.failed)} could not be moved.` : '');
      deps.saveSettings(db, workspace.id, { ...current, lastAutoTrashRun: now.toISOString(), lastAutoTrashSummary: result.trashed });
    }
    results.push(result);
  }
  return results;
}

/** Checks hourly (and a minute after start, so it never competes with startup) whether either automatic
 * step is due in any workspace. Does nothing at all while every workspace has both settings off. Returns a function that stops it. */
export function startCleanupSchedule(deps: CleanupDeps, checkEveryMs = 60 * 60 * 1000): () => void {
  let running = false;
  const check = () => {
    if (running) return;
    running = true;
    runAutoCleanupIfDue(deps)
      .catch((err) => console.error('Scheduled cleanup failed:', err))
      .finally(() => {
        running = false;
      });
  };
  const first = setTimeout(check, 60 * 1000);
  const timer = setInterval(check, checkEveryMs);
  return () => {
    clearTimeout(first);
    clearInterval(timer);
  };
}
