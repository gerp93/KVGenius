/**
 * Library cleanup: items nobody kept (not a favorite, not pinned) older than a number of days can be
 * moved to the Trash, and the Trash emptied for good. Both are manual - buttons in Settings - unless
 * the user turns on the automatic schedule, which is off by default.
 */

export interface CleanupSettings {
  /** Run the cleanup by itself, about once a day. Off unless the user turns it on. */
  autoEnabled: boolean;
  /** Items older than this many days (and not favorited or pinned) are moved to the Trash. */
  olderThanDays: number;
  /** With the schedule on, items stay in the Trash this many days before being deleted for good. */
  trashRetentionDays: number;
  /** When the schedule last ran (ISO), or null if it has not. */
  lastAutoRun: string | null;
  /** What the last scheduled run did, in words. */
  lastAutoSummary: string | null;
}

export const MIN_CLEANUP_DAYS = 1;
export const MAX_CLEANUP_DAYS = 3650;
export const DEFAULT_CLEANUP_SETTINGS: CleanupSettings = {
  autoEnabled: false,
  olderThanDays: 30,
  trashRetentionDays: 30,
  lastAutoRun: null,
  lastAutoSummary: null,
};

/** How many items, and how many bytes of files, a cleanup would move or the Trash holds. */
export interface TrashStats {
  count: number;
  bytes: number;
}

/** What moving items to the Trash did. */
export interface TrashMoveResult {
  moved: number;
  /** Left alone because it was a favorite or pinned (those are never trashed), or already in the Trash. */
  skipped: number;
  /** Could not be moved (the file is locked, on another drive, ...). */
  failed: number;
}

/** How often the schedule may run. */
export const AUTO_CLEANUP_INTERVAL_MS = 24 * 60 * 60 * 1000;

/** A whole number of days within the allowed range; anything else falls back to `fallback`. */
export function normalizeDays(value: unknown, fallback: number): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(Math.round(n), MIN_CLEANUP_DAYS), MAX_CLEANUP_DAYS);
}

/** Cleans up whatever was stored (or sent from the UI) into a valid settings object. */
export function normalizeCleanupSettings(raw: unknown): CleanupSettings {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<keyof CleanupSettings, unknown>>;
  const validDate = (v: unknown) => (typeof v === 'string' && !Number.isNaN(Date.parse(v)) ? v : null);
  return {
    autoEnabled: r.autoEnabled === true,
    olderThanDays: normalizeDays(r.olderThanDays, DEFAULT_CLEANUP_SETTINGS.olderThanDays),
    trashRetentionDays: normalizeDays(r.trashRetentionDays, DEFAULT_CLEANUP_SETTINGS.trashRetentionDays),
    lastAutoRun: validDate(r.lastAutoRun),
    lastAutoSummary: typeof r.lastAutoSummary === 'string' ? r.lastAutoSummary.slice(0, 300) : null,
  };
}

/** Applies a change made in Settings. Turning the schedule on starts its clock at that moment: the
 * first automatic run is a day later, never straight away, so nothing is moved by surprise. */
export function updateCleanupSettings(
  current: CleanupSettings,
  patch: Partial<Pick<CleanupSettings, 'autoEnabled' | 'olderThanDays' | 'trashRetentionDays'>>,
  now: Date
): CleanupSettings {
  const merged = normalizeCleanupSettings({ ...current, ...patch });
  if (merged.autoEnabled && !current.autoEnabled) merged.lastAutoRun = now.toISOString();
  return merged;
}

/** The moment `days` days before `now`, as the ISO text the database stores dates in. */
export function cutoffIso(days: number, now: Date): string {
  return new Date(now.getTime() - days * 24 * 60 * 60 * 1000).toISOString();
}

/** Whether the schedule should run now: it is on, and it has not run in the last day. */
export function autoCleanupDue(settings: CleanupSettings, now: Date): boolean {
  if (!settings.autoEnabled) return false;
  if (settings.lastAutoRun === null) return true;
  return now.getTime() - Date.parse(settings.lastAutoRun) >= AUTO_CLEANUP_INTERVAL_MS;
}
