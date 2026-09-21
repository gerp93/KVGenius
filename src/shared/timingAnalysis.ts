import { TimingStatRow } from './types';

/**
 * Crunches the timing-accuracy rows. Works only on timings and settings - the rows carry nothing
 * about prompts or images. "Difference" everywhere is (actual - estimate) / estimate: positive means
 * the run took longer than estimated, negative means it was faster.
 */

export interface AccuracySummary {
  /** Runs that had an estimate to compare with. */
  runs: number;
  /** Average size of the miss, ignoring direction, as a percentage of the estimate. */
  meanAbsPct: number;
  medianAbsPct: number;
  /** Average signed miss: positive = usually slower than estimated. */
  biasPct: number;
  within10Pct: number;
  within25Pct: number;
  avgEstimateMs: number;
  avgActualMs: number;
}

export type Metric = 'total' | 'generate';

/** Estimate/actual pair for a row under a metric; null when the row had no usable estimate. */
export function pairFor(row: TimingStatRow, metric: Metric): { estimate: number; actual: number } | null {
  const estimate = metric === 'total' ? row.estimateMs : row.estimateGenerateMs;
  const actual = metric === 'total' ? row.actualMs : row.generateMs;
  if (estimate === null || actual === null || estimate <= 0) return null;
  return { estimate, actual };
}

export function diffPct(estimate: number, actual: number): number {
  return ((actual - estimate) / estimate) * 100;
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function summarize(rows: TimingStatRow[], metric: Metric): AccuracySummary | null {
  const pairs = rows.map((r) => pairFor(r, metric)).filter((p): p is { estimate: number; actual: number } => p !== null);
  if (pairs.length === 0) return null;
  const diffs = pairs.map((p) => diffPct(p.estimate, p.actual));
  const abs = diffs.map(Math.abs);
  return {
    runs: pairs.length,
    meanAbsPct: mean(abs),
    medianAbsPct: median(abs),
    biasPct: mean(diffs),
    within10Pct: abs.filter((d) => d <= 10).length / pairs.length,
    within25Pct: abs.filter((d) => d <= 25).length / pairs.length,
    avgEstimateMs: mean(pairs.map((p) => p.estimate)),
    avgActualMs: mean(pairs.map((p) => p.actual)),
  };
}

/** Human label for the settings of a run (aspect ratio / size, steps, CFG, video length). */
export function settingsLabel(row: TimingStatRow, fps = 16): string {
  const size = `${row.width}×${row.height}`;
  if (row.kind === 'video') {
    const seconds = row.length === null ? '?' : String(Math.round(((row.length - 1) / fps) * 4) / 4);
    return `${size} · ${seconds}s video`;
  }
  return `${size} · ${row.steps} steps · CFG ${row.cfg}`;
}

export interface SettingsGroup {
  label: string;
  runs: number;
  summary: AccuracySummary | null;
}

/** Accuracy per distinct settings, most-run first. */
export function groupBySettings(rows: TimingStatRow[], metric: Metric, limit = 15): SettingsGroup[] {
  const groups = new Map<string, TimingStatRow[]>();
  for (const row of rows) {
    const key = `${row.kind}|${settingsLabel(row)}`;
    const list = groups.get(key);
    if (list) list.push(row);
    else groups.set(key, [row]);
  }
  return [...groups.entries()]
    .map(([key, list]) => ({ label: key.split('|')[1], kind: list[0].kind, runs: list.length, summary: summarize(list, metric) }))
    .sort((a, b) => b.runs - a.runs)
    .slice(0, limit)
    .map(({ label, runs, summary }) => ({ label, runs, summary }));
}

export interface TrendPoint {
  id: number;
  createdAt: string;
  diffPct: number;
}

/** The last `count` runs that had an estimate, oldest first, for plotting how the misses evolve. */
export function trend(rows: TimingStatRow[], metric: Metric, count = 60): TrendPoint[] {
  const points: TrendPoint[] = [];
  for (const row of rows) {
    const pair = pairFor(row, metric);
    if (pair) points.push({ id: row.id, createdAt: row.createdAt, diffPct: diffPct(pair.estimate, pair.actual) });
  }
  return points.slice(0, count).reverse();
}

/** Is accuracy improving? Mean miss of the newest `window` runs vs the `window` before them. */
export function recentVsEarlier(
  rows: TimingStatRow[],
  metric: Metric,
  window = 10
): { recent: number; earlier: number } | null {
  const abs: number[] = [];
  for (const row of rows) {
    const pair = pairFor(row, metric);
    if (pair) abs.push(Math.abs(diffPct(pair.estimate, pair.actual)));
  }
  if (abs.length < window * 2) return null;
  return { recent: mean(abs.slice(0, window)), earlier: mean(abs.slice(window, window * 2)) };
}
