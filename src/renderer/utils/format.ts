/** 1536 -> "1.5 KB", 5_242_880 -> "5.0 MB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

/** 75 -> "1m 15s", 8 -> "8s". */
export function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

/** A duration for people: 800 -> "0.8s", 8400 -> "8.4s", 42000 -> "42s", 95000 -> "1m 35s", 3_900_000 -> "1h 5m". */
export function formatDuration(ms: number): string {
  const abs = Math.max(0, Math.abs(ms));
  if (abs < 10_000) return `${(abs / 1000).toFixed(1)}s`;
  const totalSeconds = Math.round(abs / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  if (minutes < 60) return `${minutes}m ${String(totalSeconds % 60).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** "about 42s" - an estimate, worded as one. */
export function formatEstimate(ms: number): string {
  return `about ${formatDuration(ms)}`;
}

/**
 * How far the actual time was from the estimate, e.g. "4.0s faster (-10%)" or "12s slower (+25%)".
 * Positive difference = it took longer than estimated.
 */
export function formatDifference(estimateMs: number, actualMs: number): string {
  const diff = actualMs - estimateMs;
  if (Math.abs(diff) < 500) return 'right on the estimate';
  const pct = estimateMs > 0 ? Math.round((diff / estimateMs) * 100) : 0;
  const sign = diff > 0 ? '+' : '-';
  return `${formatDuration(diff)} ${diff > 0 ? 'slower' : 'faster'} (${sign}${Math.abs(pct)}%)`;
}
