import { DatabaseSync } from 'node:sqlite';
import { CleanupSettings, autoEmptyDue, autoTrashDue } from '../shared/cleanup';
import { Recycle, emptyTrash, runCleanup } from './trash';

/** What the scheduled cleanup needs from the app, passed in so it can run (and be tested) on its own. */
export interface CleanupDeps {
  getDb: () => DatabaseSync | null;
  getSettings: () => CleanupSettings;
  saveSettings: (settings: CleanupSettings) => void;
  trashDir: () => string;
  sourcesDir: () => string;
  /** Sends a file to the operating system's Recycle Bin. */
  recycle: Recycle;
}

function plural(n: number): string {
  return `${n} item${n === 1 ? '' : 's'}`;
}

/** What the two schedules did on this check: a sentence each, or null if that one did not run. */
export interface AutoCleanupResult {
  trashed: string | null;
  emptied: string | null;
}

/**
 * Runs whichever automatic step is on and due (each only if the user turned it on, and not more than
 * once a day):
 *  - the Trash is emptied: items that have been in it long enough go to the Recycle Bin;
 *  - old items nobody kept are moved to the Trash.
 * The two are separate options and are checked separately.
 */
export async function runAutoCleanupIfDue(deps: CleanupDeps, now: Date = new Date()): Promise<AutoCleanupResult> {
  const result: AutoCleanupResult = { trashed: null, emptied: null };
  const db = deps.getDb();
  if (!db) return result;

  const start = deps.getSettings();
  if (autoEmptyDue(start, now)) {
    const emptied = await emptyTrash(db, deps.sourcesDir(), deps.recycle, start.trashRetentionDays, now);
    result.emptied =
      `Sent ${plural(emptied.deleted)} to the Recycle Bin.` + (emptied.failed > 0 ? ` ${plural(emptied.failed)} could not be sent.` : '');
    // Settings can change while the files are being sent, so re-read before saving.
    deps.saveSettings({ ...deps.getSettings(), lastAutoEmptyRun: now.toISOString(), lastAutoEmptySummary: result.emptied });
  }

  if (autoTrashDue(deps.getSettings(), now)) {
    const settings = deps.getSettings();
    const moved = runCleanup(db, settings.olderThanDays, deps.trashDir(), now);
    result.trashed = `Moved ${plural(moved.moved)} to the Trash.` + (moved.failed > 0 ? ` ${plural(moved.failed)} could not be moved.` : '');
    deps.saveSettings({ ...settings, lastAutoTrashRun: now.toISOString(), lastAutoTrashSummary: result.trashed });
  }
  return result;
}

/** Checks hourly (and a minute after start, so it never competes with startup) whether either automatic
 * step is due. Does nothing at all while both settings are off. Returns a function that stops it. */
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
