/**
 * Deleting and cleaning up the Library is a two-step process, and nothing in it is final until the
 * last step:
 *
 *   1. Deleting an item in the app - or the cleanup, which moves items older than N days that were
 *      not favorited or pinned - only moves it to the Trash (a folder inside the app's own data).
 *      It can be restored from there.
 *   2. Emptying the Trash sends the files to the operating system's Recycle Bin.
 *
 * Both steps have an automatic schedule, each its own option and both off by default; until the user
 * turns one on, the buttons in Settings are the only way either happens.
 */

export interface CleanupSettings {
  /** Step 1 by itself, about once a day: move old items nobody kept to the Trash. Off unless turned on. */
  autoTrashEnabled: boolean;
  /** Items older than this many days (and not favorited or pinned) are moved to the Trash. */
  olderThanDays: number;
  /** Step 2 by itself, about once a day: send items that have sat in the Trash long enough to the Recycle Bin. */
  autoEmptyEnabled: boolean;
  /** An item must have been in the Trash this many days before the automatic emptying sends it on. */
  trashRetentionDays: number;
  /** When each schedule last ran (ISO), or null if it has not, and what it did, in words. */
  lastAutoTrashRun: string | null;
  lastAutoTrashSummary: string | null;
  lastAutoEmptyRun: string | null;
  lastAutoEmptySummary: string | null;
}

export const MIN_CLEANUP_DAYS = 1;
export const MAX_CLEANUP_DAYS = 3650;
export const DEFAULT_CLEANUP_SETTINGS: CleanupSettings = {
  autoTrashEnabled: false,
  olderThanDays: 30,
  autoEmptyEnabled: false,
  trashRetentionDays: 30,
  lastAutoTrashRun: null,
  lastAutoTrashSummary: null,
  lastAutoEmptyRun: null,
  lastAutoEmptySummary: null,
};

/** How many items, and how many bytes of files, a cleanup would move or the Trash holds. */
export interface TrashStats {
  count: number;
  bytes: number;
}

/** What moving items to the Trash did. */
export interface TrashMoveResult {
  moved: number;
  /** Left alone because it was a favorite or pinned and the caller protects those, or already in the Trash. */
  skipped: number;
  /** Could not be moved (the file is locked, on another drive, ...). */
  failed: number;
}

/** What emptying the Trash (sending to the Recycle Bin) did. */
export interface TrashEmptyResult {
  deleted: number;
  /** The Recycle Bin would not take the file, so the item stays in the Trash. */
  failed: number;
}

/** How often each schedule may run. */
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
  const date = (v: unknown) => (typeof v === 'string' && !Number.isNaN(Date.parse(v)) ? v : null);
  const text = (v: unknown) => (typeof v === 'string' ? v.slice(0, 300) : null);
  return {
    autoTrashEnabled: r.autoTrashEnabled === true,
    olderThanDays: normalizeDays(r.olderThanDays, DEFAULT_CLEANUP_SETTINGS.olderThanDays),
    autoEmptyEnabled: r.autoEmptyEnabled === true,
    trashRetentionDays: normalizeDays(r.trashRetentionDays, DEFAULT_CLEANUP_SETTINGS.trashRetentionDays),
    lastAutoTrashRun: date(r.lastAutoTrashRun),
    lastAutoTrashSummary: text(r.lastAutoTrashSummary),
    lastAutoEmptyRun: date(r.lastAutoEmptyRun),
    lastAutoEmptySummary: text(r.lastAutoEmptySummary),
  };
}

type SettingsPatch = Partial<Pick<CleanupSettings, 'autoTrashEnabled' | 'olderThanDays' | 'autoEmptyEnabled' | 'trashRetentionDays'>>;

/** Applies a change made in Settings. Turning a schedule on starts its clock at that moment: its first
 * automatic run is a day later, never straight away, so nothing is moved or emptied by surprise. */
export function updateCleanupSettings(current: CleanupSettings, patch: SettingsPatch, now: Date): CleanupSettings {
  const merged = normalizeCleanupSettings({ ...current, ...patch });
  if (merged.autoTrashEnabled && !current.autoTrashEnabled) merged.lastAutoTrashRun = now.toISOString();
  if (merged.autoEmptyEnabled && !current.autoEmptyEnabled) merged.lastAutoEmptyRun = now.toISOString();
  return merged;
}

/** The moment `days` days before `now`, as the ISO text the database stores dates in. */
export function cutoffIso(days: number, now: Date): string {
  return new Date(now.getTime() - days * 24 * 60 * 60 * 1000).toISOString();
}

function due(enabled: boolean, lastRun: string | null, now: Date): boolean {
  if (!enabled) return false;
  if (lastRun === null) return true;
  return now.getTime() - Date.parse(lastRun) >= AUTO_CLEANUP_INTERVAL_MS;
}

/** Whether the automatic move-to-Trash should run now: it is on, and has not run in the last day. */
export function autoTrashDue(settings: CleanupSettings, now: Date): boolean {
  return due(settings.autoTrashEnabled, settings.lastAutoTrashRun, now);
}

/** Whether the automatic emptying of the Trash should run now: it is on, and has not run in the last day. */
export function autoEmptyDue(settings: CleanupSettings, now: Date): boolean {
  return due(settings.autoEmptyEnabled, settings.lastAutoEmptyRun, now);
}
