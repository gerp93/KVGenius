import { DatabaseSync } from 'node:sqlite';
import { CleanupSettings, autoCleanupDue } from '../shared/cleanup';
import { emptyTrash, runCleanup } from './trash';

/** What the scheduled cleanup needs from the app, passed in so it can run (and be tested) on its own. */
export interface CleanupDeps {
  getDb: () => DatabaseSync | null;
  getSettings: () => CleanupSettings;
  saveSettings: (settings: CleanupSettings) => void;
  trashDir: () => string;
  sourcesDir: () => string;
}

function plural(n: number): string {
  return `${n} item${n === 1 ? '' : 's'}`;
}

/** Runs the automatic cleanup if the user turned it on and it has not run in the last day: old
 * unkept items go to the Trash, and items that have sat in the Trash long enough are deleted for good.
 * Returns what it did, or null if it did not run. */
export function runAutoCleanupIfDue(deps: CleanupDeps, now: Date = new Date()): string | null {
  const db = deps.getDb();
  if (!db) return null;
  const settings = deps.getSettings();
  if (!autoCleanupDue(settings, now)) return null;
  const deleted = emptyTrash(db, deps.sourcesDir(), settings.trashRetentionDays, now);
  const moved = runCleanup(db, settings.olderThanDays, deps.trashDir(), now);
  const summary = `Moved ${plural(moved.moved)} to the Trash and deleted ${plural(deleted)} from it for good.`;
  deps.saveSettings({ ...settings, lastAutoRun: now.toISOString(), lastAutoSummary: summary });
  return summary;
}

/** Checks hourly (and a minute after start, so it never competes with startup) whether the automatic
 * cleanup is due. Does nothing at all while the setting is off. Returns a function that stops it. */
export function startCleanupSchedule(deps: CleanupDeps, checkEveryMs = 60 * 60 * 1000): () => void {
  const check = () => {
    try {
      runAutoCleanupIfDue(deps);
    } catch (err) {
      console.error('Scheduled cleanup failed:', err);
    }
  };
  const first = setTimeout(check, 60 * 1000);
  const timer = setInterval(check, checkEveryMs);
  return () => {
    clearTimeout(first);
    clearInterval(timer);
  };
}
