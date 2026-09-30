import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { ItemView, getItem, upsertImport } from './library';
import { EndBehavior, FfmpegPaths, RunningFfmpeg, Transition, planAssemble, probeMedia, runFfmpeg } from './mediaTools';

export const ASSEMBLIES_SCHEMA = `
CREATE TABLE IF NOT EXISTS assemblies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  status TEXT NOT NULL,
  batch TEXT,
  spec TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0,
  total_seconds REAL,
  warnings TEXT NOT NULL DEFAULT '[]',
  error TEXT,
  item_id TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
`;

export type AssemblyStatus = 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';

/** An assembly request whose item ids have already been resolved to files. */
export interface AssemblySpec {
  clips: Array<{ itemId: string; path: string; seconds?: number | null }>;
  audio?: { itemId: string; path: string } | null;
  transition: Transition;
  crossfadeSeconds: number;
  end: EndBehavior;
  fadeSeconds: number;
  name?: string | null;
  batch?: string | null;
}

export interface AssemblyView {
  id: number;
  status: AssemblyStatus;
  batch: string | null;
  /** 0-1 while running. */
  progress: number;
  totalSeconds: number | null;
  warnings: string[];
  error: string | null;
  item: ItemView | null;
  createdAt: string;
  finishedAt: string | null;
}

interface AssemblyRow {
  id: number;
  status: string;
  batch: string | null;
  progress: number;
  total_seconds: number | null;
  warnings: string;
  error: string | null;
  item_id: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface AssemblyDeps {
  ffmpeg: () => FfmpegPaths | null;
  /** Where finished videos go. */
  outputDir: () => string;
  /** Scratch space for concat lists and partial output. */
  tempDir: () => string;
}

const sanitizeName = (name: string) =>
  name
    .replace(/[^A-Za-z0-9._ -]+/g, '_')
    .replace(/^\.+/, '')
    .trim()
    .slice(0, 80);

/**
 * Runs "stitch these clips and put this track under them" jobs with ffmpeg. They run in the
 * background (a re-encode of a few minutes of video takes a while), alongside generation - ffmpeg
 * does not compete with ComfyUI for the queue - and are tracked in the database like jobs are.
 */
export class AssemblyManager {
  private readonly running = new Map<number, RunningFfmpeg | 'preparing'>();
  private readonly cancelRequested = new Set<number>();
  private readonly waiters = new Map<number, Array<() => void>>();

  constructor(
    private readonly db: DatabaseSync,
    private readonly deps: AssemblyDeps
  ) {
    db.prepare(
      "UPDATE assemblies SET status = 'interrupted', error = 'KVGenius closed before this finished.', finished_at = ? WHERE status = 'running'"
    ).run(new Date().toISOString());
  }

  /** Starts an assembly in the background and returns it (status 'running'). */
  start(spec: AssemblySpec): AssemblyView {
    if (!this.deps.ffmpeg()) throw new Error('ffmpeg was not found.');
    const result = this.db
      .prepare("INSERT INTO assemblies (status, batch, spec, created_at) VALUES ('running', ?, ?, ?)")
      .run(spec.batch ?? null, JSON.stringify(spec), new Date().toISOString());
    const id = Number(result.lastInsertRowid);
    this.running.set(id, 'preparing');
    void this.execute(id, spec);
    return this.get(id) as AssemblyView;
  }

  get(id: number): AssemblyView | null {
    const row = this.db.prepare('SELECT * FROM assemblies WHERE id = ?').get(id) as unknown as AssemblyRow | undefined;
    return row ? this.toView(row) : null;
  }

  /** Resolves once the assembly has finished or `ms` has passed, whichever is first. */
  async wait(id: number, ms: number): Promise<AssemblyView | null> {
    const view = this.get(id);
    if (!view || view.status !== 'running' || ms <= 0) return view;
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, ms);
      const list = this.waiters.get(id) ?? [];
      list.push(() => {
        clearTimeout(timer);
        resolve();
      });
      this.waiters.set(id, list);
    });
    return this.get(id);
  }

  cancel(id: number): boolean {
    const active = this.running.get(id);
    if (!active) return false;
    this.cancelRequested.add(id);
    if (active !== 'preparing') active.cancel();
    return true;
  }

  private async execute(id: number, spec: AssemblySpec): Promise<void> {
    const scratch = path.join(this.deps.tempDir(), `assembly-${id}`);
    try {
      const ff = this.deps.ffmpeg();
      if (!ff) throw new Error('ffmpeg was not found.');
      fs.mkdirSync(scratch, { recursive: true });
      fs.mkdirSync(this.deps.outputDir(), { recursive: true });

      const probed = await Promise.all(spec.clips.map((c) => probeMedia(ff, c.path)));
      spec.clips.forEach((clip, i) => {
        if (!probed[i].hasVideo) throw new Error(`${clip.itemId} is not a video.`);
        if (!probed[i].durationSeconds) throw new Error(`Could not read the length of ${clip.itemId}.`);
      });
      const audioInfo = spec.audio ? await probeMedia(ff, spec.audio.path) : null;
      if (spec.audio && (!audioInfo?.hasAudio || !audioInfo.durationSeconds)) {
        throw new Error(`${spec.audio.itemId} has no readable audio.`);
      }
      if (this.cancelRequested.has(id)) throw new Error('Assembly cancelled.');

      const first = probed[0];
      const output = this.uniqueOutputPath(spec.name);
      const partial = path.join(scratch, 'output.mp4');
      const plan = planAssemble({
        clips: spec.clips.map((c, i) => ({ path: c.path, durationSeconds: probed[i].durationSeconds as number, trimSeconds: c.seconds })),
        width: first.width ?? 640,
        height: first.height ?? 640,
        fps: first.fps ?? 16,
        audio: spec.audio && audioInfo ? { path: spec.audio.path, durationSeconds: audioInfo.durationSeconds as number } : null,
        transition: spec.transition,
        crossfadeSeconds: spec.crossfadeSeconds,
        end: spec.end,
        fadeSeconds: spec.fadeSeconds,
        output: partial,
        listFilePath: path.join(scratch, 'clips.txt'),
      });
      if (plan.listFileContent !== undefined) fs.writeFileSync(path.join(scratch, 'clips.txt'), plan.listFileContent);
      this.db
        .prepare('UPDATE assemblies SET total_seconds = ?, warnings = ? WHERE id = ?')
        .run(plan.totalSeconds, JSON.stringify(plan.warnings), id);

      let lastWritten = 0;
      const running = runFfmpeg(ff, plan.args, plan.totalSeconds, (fraction) => {
        if (fraction - lastWritten >= 0.02) {
          lastWritten = fraction;
          this.db.prepare('UPDATE assemblies SET progress = ? WHERE id = ?').run(fraction, id);
        }
      });
      this.running.set(id, running);
      if (this.cancelRequested.has(id)) running.cancel();
      await running.done;

      fs.renameSync(partial, output);
      const info = await probeMedia(ff, output).catch(() => null);
      const item = upsertImport(this.db, {
        path: output,
        kind: 'video',
        origin: 'assembled',
        width: info?.width,
        height: info?.height,
        duration: info?.durationSeconds ?? plan.totalSeconds,
        batch: spec.batch,
      });
      this.db
        .prepare("UPDATE assemblies SET status = 'done', progress = 1, item_id = ?, finished_at = ? WHERE id = ?")
        .run(item.id, new Date().toISOString(), id);
    } catch (err) {
      const cancelled = this.cancelRequested.has(id);
      this.db
        .prepare('UPDATE assemblies SET status = ?, error = ?, finished_at = ? WHERE id = ?')
        .run(cancelled ? 'cancelled' : 'failed', cancelled ? null : err instanceof Error ? err.message : String(err), new Date().toISOString(), id);
    } finally {
      this.running.delete(id);
      this.cancelRequested.delete(id);
      fs.rmSync(scratch, { recursive: true, force: true });
      const waiting = this.waiters.get(id);
      this.waiters.delete(id);
      waiting?.forEach((resolve) => resolve());
    }
  }

  private uniqueOutputPath(name?: string | null): string {
    const base = (name ? sanitizeName(name) : '') || `assembled-${Date.now()}`;
    let candidate = path.join(this.deps.outputDir(), `${base}.mp4`);
    for (let n = 2; fs.existsSync(candidate); n++) candidate = path.join(this.deps.outputDir(), `${base}-${n}.mp4`);
    return candidate;
  }

  private toView(row: AssemblyRow): AssemblyView {
    return {
      id: row.id,
      status: row.status as AssemblyStatus,
      batch: row.batch,
      progress: Math.round(row.progress * 1000) / 1000,
      totalSeconds: row.total_seconds,
      warnings: JSON.parse(row.warnings) as string[],
      error: row.error,
      item: row.item_id ? getItem(this.db, row.item_id) : null,
      createdAt: row.created_at,
      finishedAt: row.finished_at,
    };
  }
}
