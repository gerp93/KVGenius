import { DatabaseSync } from 'node:sqlite';
import { TimingStatRow } from '../shared/types';

/**
 * The timing-accuracy data, kept apart from the generations it describes: no prompt, seed, image
 * or file path, and no reference back to the generation. Deleting a generation therefore leaves
 * only these numbers behind (which is the point - they exist to show how good the estimates are).
 */
export const TIMING_SCHEMA = `
CREATE TABLE IF NOT EXISTS timing_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  family TEXT NOT NULL,
  kind TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  steps INTEGER NOT NULL,
  cfg REAL NOT NULL,
  length INTEGER,
  warm INTEGER NOT NULL DEFAULT 0,
  estimate_ms INTEGER,
  estimate_generate_ms INTEGER,
  actual_ms INTEGER NOT NULL,
  load_ms INTEGER,
  generate_ms INTEGER,
  sampling_ms INTEGER,
  finish_ms INTEGER,
  sampler_steps INTEGER,
  pace_ms REAL
);
`;

interface TimingRow {
  id: number;
  created_at: string;
  family: string;
  kind: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  length: number | null;
  warm: number;
  estimate_ms: number | null;
  estimate_generate_ms: number | null;
  actual_ms: number;
  load_ms: number | null;
  generate_ms: number | null;
  sampling_ms: number | null;
  finish_ms: number | null;
  sampler_steps: number | null;
  pace_ms: number | null;
}

function rowToTiming(r: TimingRow): TimingStatRow {
  return {
    id: r.id,
    createdAt: r.created_at,
    family: r.family,
    kind: r.kind === 'video' ? 'video' : 'image',
    width: r.width,
    height: r.height,
    steps: r.steps,
    cfg: r.cfg,
    length: r.length,
    warm: r.warm === 1,
    estimateMs: r.estimate_ms,
    estimateGenerateMs: r.estimate_generate_ms,
    actualMs: r.actual_ms,
    loadMs: r.load_ms,
    generateMs: r.generate_ms,
    samplingMs: r.sampling_ms,
    finishMs: r.finish_ms,
    samplerSteps: r.sampler_steps,
    paceMs: r.pace_ms,
  };
}

const round = (n: number | null): number | null => (n === null ? null : Math.round(n));

/** Stores the timing of one finished generation and returns its id. */
export function insertTiming(db: DatabaseSync, t: Omit<TimingStatRow, 'id' | 'createdAt'>): number {
  const result = db
    .prepare(
      `INSERT INTO timing_stats (created_at, family, kind, width, height, steps, cfg, length, warm,
         estimate_ms, estimate_generate_ms, actual_ms, load_ms, generate_ms, sampling_ms, finish_ms, sampler_steps, pace_ms)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      new Date().toISOString(),
      t.family,
      t.kind,
      t.width,
      t.height,
      t.steps,
      t.cfg,
      t.length,
      t.warm ? 1 : 0,
      round(t.estimateMs),
      round(t.estimateGenerateMs),
      Math.round(t.actualMs),
      round(t.loadMs),
      round(t.generateMs),
      round(t.samplingMs),
      round(t.finishMs),
      t.samplerSteps,
      t.paceMs
    );
  return Number(result.lastInsertRowid);
}

/** Every timing row, newest first (the accuracy page and the estimator both read these). */
export function listTimingRows(db: DatabaseSync, limit?: number): TimingStatRow[] {
  const rows = (
    limit === undefined
      ? db.prepare('SELECT * FROM timing_stats ORDER BY id DESC').all()
      : db.prepare('SELECT * FROM timing_stats ORDER BY id DESC LIMIT ?').all(limit)
  ) as unknown as TimingRow[];
  return rows.map(rowToTiming);
}

export function clearTimingStats(db: DatabaseSync): void {
  db.exec('DELETE FROM timing_stats');
  db.exec('UPDATE generations SET timing_id = NULL');
}
