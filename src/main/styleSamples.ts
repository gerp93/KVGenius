import { DatabaseSync } from 'node:sqlite';
import { GenerationParams, GenerationRecord } from '../shared/types';
import { JobInfo, JobRequest } from '../shared/jobs';
import { Z_IMAGE_FAMILY } from '../shared/families';
import { PromptStyle, combinePrompt } from '../shared/styles';
import {
  BASELINE_ID,
  DEFAULT_SAMPLE_SETTINGS,
  SAMPLE_CFG,
  SAMPLE_SIZE,
  SAMPLE_STEPS,
  SampleSnapshot,
  SampleState,
  StyleSampleInfo,
  StyleSampleSettings,
  StyleSamplesView,
  sampleFreshness,
  validateSampleSettings,
} from '../shared/styleSamples';

/**
 * The Styles page's example pictures (see shared/styleSamples.ts). Each is an ordinary Library item - made through the
 * normal queue, hidden from the Library's lists unless "Show hidden" is on - that this table points at, with the wording and
 * standard prompt it was made from, so a later change shows up as "outdated".
 */
export const STYLE_SAMPLES_SCHEMA = `
CREATE TABLE IF NOT EXISTS style_sample_settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS style_samples (
  style_id INTEGER PRIMARY KEY,
  generation_id INTEGER NOT NULL,
  text_snapshot TEXT NOT NULL,
  prompt_snapshot TEXT NOT NULL,
  seed_snapshot INTEGER NOT NULL,
  rendered_at TEXT NOT NULL
);
`;

export function getSampleSettings(db: DatabaseSync): StyleSampleSettings {
  const rows = db.prepare('SELECT key, value FROM style_sample_settings').all() as unknown as { key: string; value: string }[];
  const stored = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  const checked = validateSampleSettings({ prompt: stored.prompt ?? DEFAULT_SAMPLE_SETTINGS.prompt, seed: stored.seed ?? DEFAULT_SAMPLE_SETTINGS.seed });
  return checked.ok ? checked.value : DEFAULT_SAMPLE_SETTINGS;
}

/** Saves the standard prompt and seed. Throws a message fit to show the user. */
export function saveSampleSettings(db: DatabaseSync, input: { prompt?: unknown; seed?: unknown }): StyleSampleSettings {
  const checked = validateSampleSettings(input);
  if (!checked.ok) throw new Error(checked.message);
  const put = db.prepare('INSERT INTO style_sample_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value');
  put.run('prompt', checked.value.prompt);
  put.run('seed', String(checked.value.seed));
  return checked.value;
}

interface SampleRow {
  style_id: number;
  generation_id: number;
  text_snapshot: string;
  prompt_snapshot: string;
  seed_snapshot: number;
  rendered_at: string;
}

export interface StoredSample {
  styleId: number;
  generationId: number;
  snapshot: SampleSnapshot;
  renderedAt: string;
}

function rowToSample(row: SampleRow): StoredSample {
  return {
    styleId: row.style_id,
    generationId: row.generation_id,
    snapshot: { text: row.text_snapshot, prompt: row.prompt_snapshot, seed: row.seed_snapshot },
    renderedAt: row.rendered_at,
  };
}

export function getStoredSample(db: DatabaseSync, styleId: number): StoredSample | null {
  const row = db.prepare('SELECT * FROM style_samples WHERE style_id = ?').get(styleId) as unknown as SampleRow | undefined;
  return row ? rowToSample(row) : null;
}

/** Points a style's example at a finished picture. Returns the picture it replaced, if any. */
export function putStoredSample(db: DatabaseSync, styleId: number, generationId: number, snapshot: SampleSnapshot, renderedAt: string): number | null {
  const before = getStoredSample(db, styleId);
  db.prepare(
    `INSERT INTO style_samples (style_id, generation_id, text_snapshot, prompt_snapshot, seed_snapshot, rendered_at) VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(style_id) DO UPDATE SET generation_id = excluded.generation_id, text_snapshot = excluded.text_snapshot,
       prompt_snapshot = excluded.prompt_snapshot, seed_snapshot = excluded.seed_snapshot, rendered_at = excluded.rendered_at`
  ).run(styleId, generationId, snapshot.text, snapshot.prompt, snapshot.seed, renderedAt);
  return before && before.generationId !== generationId ? before.generationId : null;
}

/** Forgets a style's example (the style was deleted). Returns the picture it pointed at, if any. */
export function deleteStoredSample(db: DatabaseSync, styleId: number): number | null {
  const before = getStoredSample(db, styleId);
  db.prepare('DELETE FROM style_samples WHERE style_id = ?').run(styleId);
  return before?.generationId ?? null;
}

/** The request that makes one example: the standard prompt with the style's or element's wording added, at the fixed seed. */
export function sampleRequest(settings: StyleSampleSettings, style: PromptStyle | null): JobRequest {
  const params: GenerationParams = {
    prompt: combinePrompt(settings.prompt, style ? style.text : null),
    width: SAMPLE_SIZE,
    height: SAMPLE_SIZE,
    seed: settings.seed,
    steps: SAMPLE_STEPS,
    cfg: SAMPLE_CFG,
    ...(style ? { styleName: style.name } : {}),
  };
  return { family: Z_IMAGE_FAMILY, params, source: 'ui', batch: STYLE_SAMPLES_BATCH };
}

/** The queue batch label of every example job, so they are recognisable in the queue. */
export const STYLE_SAMPLES_BATCH = 'style-examples';

export interface StyleSamplesQueue {
  submit(request: JobRequest): JobInfo;
  get(id: number): JobInfo | null;
  wait(id: number): Promise<JobInfo>;
}

export interface StyleSamplesDeps {
  db: DatabaseSync;
  queue: StyleSamplesQueue;
  /** Sends a picture to the Trash (the one an example replaces). */
  trash: (ids: number[]) => void;
  /** Hides a new example from the Library's lists. */
  hide: (id: number) => void;
  getRecord: (id: number) => GenerationRecord | null;
  imageUrlFor: (imagePath: string) => string;
  listStyles: () => PromptStyle[];
  getStyle: (id: number) => PromptStyle | null;
}

/** Makes the examples through the queue and says how each stands. Keeps no state of its own beyond the jobs in flight. */
export class StyleSampleService {
  /** Style id -> the job making its example right now. */
  private readonly pending = new Map<number, number>();
  private readonly failures = new Map<number, string>();

  constructor(private readonly deps: StyleSamplesDeps) {}

  view(): StyleSamplesView {
    const settings = getSampleSettings(this.deps.db);
    const entries: Array<{ id: number; text: string }> = [{ id: BASELINE_ID, text: '' }, ...this.deps.listStyles().map((s) => ({ id: s.id, text: s.text }))];
    return { settings, samples: entries.map((entry) => this.info(entry.id, entry.text, settings)) };
  }

  private info(id: number, text: string, settings: StyleSampleSettings): StyleSampleInfo {
    const stored = getStoredSample(this.deps.db, id);
    const record = stored ? this.deps.getRecord(stored.generationId) : null;
    // A picture that was deleted or sent to the Trash is no example any more.
    const alive = record !== null && record.trashedAt === null;
    const freshness = alive && stored ? sampleFreshness(stored.snapshot, text, settings) : 'none';
    const jobId = this.pending.get(id);
    const job = jobId === undefined ? null : this.deps.queue.get(jobId);
    let state: SampleState = freshness;
    if (job && (job.status === 'queued' || job.status === 'running')) state = job.status === 'queued' ? 'queued' : 'rendering';
    else if (this.failures.has(id) && freshness !== 'current') state = 'failed';
    return {
      id,
      state,
      imageUrl: alive && record ? this.deps.imageUrlFor(record.imagePath) : null,
      renderedAt: alive && stored ? stored.renderedAt : null,
      error: state === 'failed' ? (this.failures.get(id) ?? null) : null,
    };
  }

  /** Queues the examples for these ids (BASELINE_ID, or a style's or element's id). One already waiting or running is left alone. */
  render(ids: number[]): number {
    const settings = getSampleSettings(this.deps.db);
    let queued = 0;
    for (const id of new Set(ids)) {
      const style = id === BASELINE_ID ? null : this.deps.getStyle(id);
      if (id !== BASELINE_ID && !style) continue;
      const running = this.pending.get(id);
      const runningJob = running === undefined ? null : this.deps.queue.get(running);
      if (runningJob && (runningJob.status === 'queued' || runningJob.status === 'running')) continue;
      const job = this.deps.queue.submit(sampleRequest(settings, style));
      this.pending.set(id, job.id);
      this.failures.delete(id);
      queued++;
      const snapshot: SampleSnapshot = { text: style ? style.text : '', prompt: settings.prompt, seed: settings.seed };
      void this.deps.queue.wait(job.id).then((done) => this.finished(id, job.id, done, snapshot));
    }
    return queued;
  }

  private finished(id: number, jobId: number, job: JobInfo, snapshot: SampleSnapshot): void {
    if (this.pending.get(id) === jobId) this.pending.delete(id);
    if (job.status === 'cancelled' || job.status === 'interrupted') return;
    if (job.status !== 'done' || job.generationId === null) {
      this.failures.set(id, job.error ?? 'The example could not be made.');
      return;
    }
    // A style deleted while its example was being made has nowhere to keep it.
    if (id !== BASELINE_ID && !this.deps.getStyle(id)) {
      this.deps.trash([job.generationId]);
      return;
    }
    this.deps.hide(job.generationId);
    const replaced = putStoredSample(this.deps.db, id, job.generationId, snapshot, new Date().toISOString());
    if (replaced !== null) this.deps.trash([replaced]);
  }

  /** A style or element was deleted: its example goes to the Trash with it. */
  forget(styleId: number): void {
    this.pending.delete(styleId);
    this.failures.delete(styleId);
    const generationId = deleteStoredSample(this.deps.db, styleId);
    if (generationId !== null) this.deps.trash([generationId]);
  }
}
