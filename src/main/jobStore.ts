import { DatabaseSync } from 'node:sqlite';
import { GenerationParams } from '../shared/types';
import { JobFilter, JobInfo, JobRequest, JobSource, JobStatus } from '../shared/jobs';

/**
 * Every generation request, whichever way it arrived, so a client that disconnects (or the app
 * restarting) does not lose track of it. The finished result lives in `generations`; a job only
 * points at it, and a deleted generation just leaves the pointer dangling.
 */
export const JOBS_SCHEMA = `
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  batch TEXT,
  family TEXT NOT NULL,
  params TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  generation_id INTEGER,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs (batch);
`;

interface JobRow {
  id: number;
  source: string;
  batch: string | null;
  family: string;
  params: string;
  status: string;
  error: string | null;
  generation_id: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

function rowToJob(row: JobRow): JobInfo {
  return {
    id: row.id,
    source: row.source as JobSource,
    batch: row.batch,
    family: row.family,
    params: JSON.parse(row.params) as GenerationParams,
    status: row.status as JobStatus,
    error: row.error,
    generationId: row.generation_id,
    createdAt: row.created_at,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
  };
}

export function insertJob(db: DatabaseSync, request: JobRequest): JobInfo {
  const result = db
    .prepare(
      `INSERT INTO jobs (source, batch, family, params, status, created_at)
       VALUES (?, ?, ?, ?, 'queued', ?)`
    )
    .run(request.source, request.batch ?? null, request.family, JSON.stringify(request.params), new Date().toISOString());
  return getJob(db, Number(result.lastInsertRowid)) as JobInfo;
}

export function getJob(db: DatabaseSync, id: number): JobInfo | null {
  const row = db.prepare('SELECT * FROM jobs WHERE id = ?').get(id) as unknown as JobRow | undefined;
  return row ? rowToJob(row) : null;
}

/** Newest first. */
export function listJobs(db: DatabaseSync, filter: JobFilter = {}): JobInfo[] {
  const where: string[] = [];
  const args: (string | number)[] = [];
  if (filter.batch !== undefined) {
    where.push('batch = ?');
    args.push(filter.batch);
  }
  if (filter.status !== undefined) {
    where.push('status = ?');
    args.push(filter.status);
  }
  const limit = Math.min(Math.max(Math.floor(filter.limit ?? 100) || 0, 1), 500);
  const sql = `SELECT * FROM jobs ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY id DESC LIMIT ${limit}`;
  return (db.prepare(sql).all(...args) as unknown as JobRow[]).map(rowToJob);
}

/** The oldest job still waiting, i.e. what runs next. */
export function nextQueuedJob(db: DatabaseSync): JobInfo | null {
  const row = db.prepare("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").get() as unknown as
    | JobRow
    | undefined;
  return row ? rowToJob(row) : null;
}

export function countQueuedJobs(db: DatabaseSync): number {
  const row = db.prepare("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'").get() as unknown as { n: number };
  return row.n;
}

export function markJobRunning(db: DatabaseSync, id: number): void {
  db.prepare("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?").run(new Date().toISOString(), id);
}

export function finishJob(
  db: DatabaseSync,
  id: number,
  status: 'done' | 'failed' | 'cancelled',
  detail: { error?: string | null; generationId?: number | null } = {}
): void {
  db.prepare('UPDATE jobs SET status = ?, error = ?, generation_id = ?, finished_at = ? WHERE id = ?').run(
    status,
    detail.error ?? null,
    detail.generationId ?? null,
    new Date().toISOString(),
    id
  );
}

/** Jobs that were queued or running when the app last closed can never finish now. */
export function markInterruptedJobs(db: DatabaseSync): number {
  const result = db
    .prepare(
      `UPDATE jobs SET status = 'interrupted', error = 'KVGenius closed before this finished.', finished_at = ?
       WHERE status IN ('queued', 'running')`
    )
    .run(new Date().toISOString());
  return Number(result.changes);
}
