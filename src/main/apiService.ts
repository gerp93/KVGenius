import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { FAMILY_KIND, GenerationParams, GenerationProgress } from '../shared/types';
import { isUpscaleFamily } from '../shared/upscale';
import { JobFilter, JobInfo, JobStatus, isTerminalJobStatus } from '../shared/jobs';
import { TOOLS, ToolResult } from '../shared/tools';
import { JobQueue } from './jobQueue';
import { AssemblyManager, AssemblySpec } from './assembly';
import { FfmpegPaths, MediaInfo, EndBehavior, Transition, makePoster, probeMedia } from './mediaTools';
import { combinePrompt } from '../shared/styles';
import { findStyleByName, listStyles } from './styles';
import { ItemKind, ItemOrigin, ItemView, getItem, kindForExtension, listItems, upsertImport } from './library';

/** A problem with the request itself (or the machine's state), reported to the client as-is. */
export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string
  ) {
    super(message);
  }
}

export interface ApiServiceDeps {
  db: DatabaseSync;
  queue: JobQueue;
  assemblies: AssemblyManager;
  ffmpeg: () => FfmpegPaths | null;
  comfyAvailable: () => Promise<boolean>;
  /** Pixel size of an image file, without needing ffmpeg. */
  imageSize: (file: string) => { width: number; height: number } | null;
  /** A JPEG preview of an image for when ffmpeg is not available. */
  imagePreviewFallback: (file: string) => Promise<Buffer | null>;
}

const FAMILIES: Record<string, { kind: 'image' | 'video'; description: string }> = {
  'z-image-turbo': { kind: 'image', description: 'Fast text-to-image (Z Image Turbo).' },
  'wan22-i2v': { kind: 'video', description: 'Image-to-video (Wan 2.2): animates a source image into a short clip at 16 fps.' },
};

const VIDEO_FPS = 16;
const VIDEO_LONG_SIDE = 640;
const MAX_IMPORT_FILES = 500;
const MAX_WAIT_SECONDS = 60;
const ITEM_ID_HINT = 'an item id such as "gen-12" or "imp-5"';

type Args = Record<string, unknown>;

// -- argument helpers ----------------------------------------------------------------------------

function fail(message: string): never {
  throw new ApiError('invalid_argument', message);
}

function reqString(args: Args, key: string, max = 20_000): string {
  const v = args[key];
  if (typeof v !== 'string' || v.trim() === '') fail(`"${key}" is required and must be a non-empty string.`);
  if (v.length > max) fail(`"${key}" is too long (max ${max} characters).`);
  return v;
}

function optString(args: Args, key: string, max = 20_000): string | undefined {
  if (args[key] === undefined || args[key] === null) return undefined;
  return reqString(args, key, max);
}

function optNumber(args: Args, key: string, min: number, max: number, integer = false): number | undefined {
  const v = args[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== 'number' || !Number.isFinite(v)) fail(`"${key}" must be a number.`);
  if (integer && !Number.isInteger(v)) fail(`"${key}" must be an integer.`);
  if (v < min || v > max) fail(`"${key}" must be between ${min} and ${max}.`);
  return v;
}

function optBool(args: Args, key: string, fallback: boolean): boolean {
  const v = args[key];
  if (v === undefined || v === null) return fallback;
  if (typeof v !== 'boolean') fail(`"${key}" must be true or false.`);
  return v;
}

function optEnum<T extends string>(args: Args, key: string, allowed: readonly T[]): T | undefined {
  const v = args[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== 'string' || !allowed.includes(v as T)) fail(`"${key}" must be one of: ${allowed.join(', ')}.`);
  return v as T;
}

function optBatch(args: Args): string | undefined {
  return optString(args, 'batch', 64)?.trim();
}

const snap = (n: number, multiple: number, min: number, max: number) =>
  Math.min(max, Math.max(min, Math.round(n / multiple) * multiple));

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function expandHome(p: string): string {
  return p === '~' || p.startsWith('~/') || p.startsWith('~\\') ? path.join(os.homedir(), p.slice(1)) : p;
}

function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

// ------------------------------------------------------------------------------------------------

export class ApiService {
  private readonly progress = new Map<number, GenerationProgress>();

  constructor(private readonly deps: ApiServiceDeps) {
    deps.queue.onProgress((jobId, progress) => this.progress.set(jobId, progress));
    deps.queue.onJobChanged((job) => {
      if (isTerminalJobStatus(job.status)) this.progress.delete(job.id);
    });
  }

  // Everything a client can see or touch is scoped to what clients themselves created: their
  // jobs, and the library items those (and their imports/assemblies) produced. Work done in the
  // app is not merely hidden from listings - it is indistinguishable from an id that does not exist.

  private item(id: string): ItemView | null {
    return getItem(this.deps.db, id, { clientOnly: true });
  }

  private clientJob(id: number): JobInfo | null {
    const job = this.deps.queue.get(id);
    return job && job.source === 'mcp' ? job : null;
  }

  async callTool(name: string, rawArgs: unknown): Promise<ToolResult> {
    if (!TOOLS.some((t) => t.name === name)) throw new ApiError('unknown_tool', `Unknown tool "${name}".`);
    if (rawArgs !== undefined && rawArgs !== null && (typeof rawArgs !== 'object' || Array.isArray(rawArgs))) {
      fail('Tool arguments must be a JSON object.');
    }
    const args = (rawArgs ?? {}) as Args;
    switch (name) {
      case 'list_capabilities':
        return { data: await this.listCapabilities() };
      case 'import_folder':
        return { data: await this.importFolder(args) };
      case 'list_styles':
        return { data: this.listStylesTool() };
      case 'generate_image':
        return { data: this.generateImage(args) };
      case 'generate_video':
        return { data: this.generateVideo(args) };
      case 'list_jobs':
        return { data: this.listJobs(args) };
      case 'get_job':
        return { data: await this.getJob(args) };
      case 'cancel_job':
        return { data: await this.cancelJob(args) };
      case 'list_library':
        return { data: this.listLibrary(args) };
      case 'get_item':
        return this.getItemTool(args);
      case 'probe_media':
        return { data: await this.probeMediaTool(args) };
      case 'assemble_video':
        return { data: await this.assembleVideo(args) };
      case 'get_assembly':
        return { data: await this.getAssembly(args) };
    }
    throw new ApiError('unknown_tool', `Unknown tool "${name}".`);
  }

  // -- capabilities ------------------------------------------------------------------------------

  private async listCapabilities() {
    const [comfyReachable] = await Promise.all([this.deps.comfyAvailable()]);
    const ff = this.deps.ffmpeg();
    return {
      comfyui_reachable: comfyReachable,
      comfyui_note: comfyReachable ? null : 'ComfyUI is not reachable. Generation jobs will fail until it is running (check the address in KVGenius Settings).',
      ffmpeg_available: ff !== null,
      ffmpeg_note: ff ? null : 'ffmpeg was not found: previews of videos, probe_media and assemble_video are unavailable. Set its location in KVGenius Settings.',
      families: [
        {
          family: 'z-image-turbo',
          ...FAMILIES['z-image-turbo'],
          tool: 'generate_image',
          fields: {
            prompt: 'text',
            style: 'optional name of a saved style (see list_styles), added after the prompt',
            width: { min: 256, max: 2048, multiple_of: 64, default: 1024 },
            height: { min: 256, max: 2048, multiple_of: 64, default: 1024 },
            seed: 'integer, random by default',
            steps: { min: 1, max: 20, default: 8 },
            cfg: { min: 0.5, max: 3, default: 1 },
          },
        },
        {
          family: 'wan22-i2v',
          ...FAMILIES['wan22-i2v'],
          tool: 'generate_video',
          fields: {
            prompt: 'text describing the motion',
            source: 'library image id (required)',
            seconds: { min: 1, max: 12, default: 5, note: 'a clip is ~5s; longer takes proportionally longer' },
            width: { min: 256, max: 1280, multiple_of: 16, default: 'from source aspect ratio, ~640 long side' },
            height: { min: 256, max: 1280, multiple_of: 16, default: 'from source aspect ratio, ~640 long side' },
            seed: 'integer, random by default',
          },
        },
      ],
      jobs_run_one_at_a_time: true,
    };
  }

  // -- importing ---------------------------------------------------------------------------------

  private async importFolder(args: Args) {
    const target = expandHome(reqString(args, 'path', 4096).trim());
    if (!path.isAbsolute(target)) fail('"path" must be an absolute path.');
    const recursive = optBool(args, 'recursive', false);
    const batch = optBatch(args);

    let stat: fs.Stats;
    try {
      stat = fs.statSync(target);
    } catch {
      throw new ApiError('not_found', `Nothing at ${target} (the path is resolved on the machine running KVGenius).`);
    }

    const files: string[] = [];
    if (stat.isDirectory()) {
      const walk = (dir: string, depth: number) => {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
          const full = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            if (recursive && depth < 4 && !entry.name.startsWith('.')) walk(full, depth + 1);
          } else if (entry.isFile() && kindForExtension(entry.name)) {
            files.push(full);
          }
        }
      };
      walk(target, 0);
    } else if (kindForExtension(target)) {
      files.push(target);
    } else {
      fail(`${path.basename(target)} is not a supported image, video or audio file.`);
    }
    if (files.length === 0) throw new ApiError('no_media', `No image, video or audio files found in ${target}.`);

    files.sort((a, b) => naturalCompare(path.relative(target, a), path.relative(target, b)));
    const truncated = files.length > MAX_IMPORT_FILES;
    const ff = this.deps.ffmpeg();
    const items: ItemView[] = [];
    for (const file of files.slice(0, MAX_IMPORT_FILES)) {
      const kind = kindForExtension(file) as ItemKind;
      let width: number | null = null;
      let height: number | null = null;
      let duration: number | null = null;
      if (kind === 'image') {
        const size = this.deps.imageSize(file);
        width = size?.width ?? null;
        height = size?.height ?? null;
      }
      if (ff && kind !== 'image') {
        const info = await probeMedia(ff, file).catch(() => null);
        width = info?.width ?? null;
        height = info?.height ?? null;
        duration = info?.durationSeconds ?? null;
      }
      items.push(upsertImport(this.deps.db, { path: file, kind, origin: 'imported', width, height, duration, batch }));
    }
    return { imported: items.length, truncated, limit: truncated ? MAX_IMPORT_FILES : undefined, items };
  }

  // -- generating --------------------------------------------------------------------------------

  private generateImage(args: Args) {
    const prompt = reqString(args, 'prompt');
    const family = optString(args, 'family', 64) ?? 'z-image-turbo';
    if (FAMILY_KIND[family] !== 'image') fail(`"${family}" is not an image family. Image families: ${Object.keys(FAMILY_KIND).filter((f) => FAMILY_KIND[f] === 'image').join(', ')}.`);
    // A style's words are added here, so the job (and the Library record) holds the full prompt that is
    // sent. With no style the prompt is passed through untouched.
    const styleName = optString(args, 'style', 200);
    const style = styleName === undefined ? null : findStyleByName(this.deps.db, styleName);
    if (styleName !== undefined && !style) {
      const names = listStyles(this.deps.db).map((s) => `"${s.name}"`);
      throw new ApiError(
        'not_found',
        `No style named "${styleName.trim()}". ${names.length ? `Saved styles: ${names.join(', ')}.` : 'There are no saved styles yet - they are created in the KVGenius Styles tab.'}`
      );
    }
    const params: GenerationParams = {
      prompt: combinePrompt(prompt, style?.text),
      ...(style ? { styleName: style.name } : {}),
      width: snap(optNumber(args, 'width', 64, 8192, true) ?? 1024, 64, 256, 2048),
      height: snap(optNumber(args, 'height', 64, 8192, true) ?? 1024, 64, 256, 2048),
      seed: optNumber(args, 'seed', 0, 2 ** 32 - 1, true) ?? Math.floor(Math.random() * 2 ** 32),
      steps: optNumber(args, 'steps', 1, 20, true) ?? 8,
      cfg: optNumber(args, 'cfg', 0.5, 3) ?? 1,
    };
    return this.jobView(this.deps.queue.submit({ family, params, source: 'mcp', batch: optBatch(args) }));
  }

  private listStylesTool() {
    const styles = listStyles(this.deps.db).map((s) => ({ name: s.name, text: s.text }));
    return {
      styles,
      note: styles.length
        ? 'Pass a name as `style` to generate_image. Its text is appended to your prompt after a comma.'
        : 'No styles saved yet - the user creates them in the KVGenius Styles tab.',
    };
  }

  private generateVideo(args: Args) {
    const prompt = reqString(args, 'prompt');
    const family = optString(args, 'family', 64) ?? 'wan22-i2v';
    if (FAMILY_KIND[family] !== 'video' || isUpscaleFamily(family)) fail(`"${family}" is not a video family. Video families: ${Object.keys(FAMILY_KIND).filter((f) => FAMILY_KIND[f] === 'video' && !isUpscaleFamily(f)).join(', ')}.`);
    const source = this.requireItem(reqString(args, 'source', 64), ['image'], 'source');

    const size = source.width && source.height ? { width: source.width, height: source.height } : this.deps.imageSize(source.path);
    let width = optNumber(args, 'width', 16, 8192, true);
    let height = optNumber(args, 'height', 16, 8192, true);
    if (width === undefined || height === undefined) {
      if (!size) throw new ApiError('unreadable_image', `Could not read the size of ${source.id}; pass width and height explicitly.`);
      // Same rule as the Generate page: keep the aspect ratio at ~640 on the long side.
      const scale = VIDEO_LONG_SIDE / Math.max(size.width, size.height);
      width = width ?? Math.max(256, Math.round((size.width * scale) / 16) * 16);
      height = height ?? Math.max(256, Math.round((size.height * scale) / 16) * 16);
    }
    const seconds = optNumber(args, 'seconds', 1, 12) ?? 5;
    const params: GenerationParams = {
      prompt,
      width: snap(width, 16, 256, 1280),
      height: snap(height, 16, 256, 1280),
      seed: optNumber(args, 'seed', 0, 2 ** 32 - 1, true) ?? Math.floor(Math.random() * 2 ** 32),
      // The video template ignores these; they match the Generate page's defaults so timing history stays comparable.
      steps: 8,
      cfg: 1,
      length: 4 * Math.round(seconds * (VIDEO_FPS / 4)) + 1,
      sourceImagePath: source.path,
    };
    return this.jobView(this.deps.queue.submit({ family, params, source: 'mcp', batch: optBatch(args) }));
  }

  // -- jobs --------------------------------------------------------------------------------------

  private jobView(job: JobInfo) {
    const p = this.progress.get(job.id);
    return {
      job_id: job.id,
      status: job.status,
      kind: FAMILY_KIND[job.family] ?? null,
      family: job.family,
      batch: job.batch,
      prompt: job.params.prompt,
      style: job.params.styleName ?? null,
      seed: job.params.seed,
      width: job.params.width,
      height: job.params.height,
      seconds: job.params.length ? Math.round(((job.params.length - 1) / VIDEO_FPS) * 100) / 100 : null,
      source_image: job.params.sourceImagePath ?? null,
      progress: p
        ? {
            phase: p.phase,
            step: p.stepsDone,
            steps_total: p.stepsTotal,
            percent: p.stepsTotal ? Math.round((p.stepsDone / p.stepsTotal) * 100) : null,
            elapsed_seconds: Math.round(p.elapsedMs / 1000),
          }
        : null,
      error: job.error,
      item: job.generationId === null ? null : this.item(`gen-${job.generationId}`),
      created_at: job.createdAt,
      started_at: job.startedAt,
      finished_at: job.finishedAt,
    };
  }

  private listJobs(args: Args) {
    const filter: JobFilter = {
      source: 'mcp',
      batch: optBatch(args),
      status: optEnum<JobStatus>(args, 'status', ['queued', 'running', 'done', 'failed', 'cancelled', 'interrupted']),
      limit: optNumber(args, 'limit', 1, 500, true),
    };
    const jobs = this.deps.queue.list(filter);
    const counts: Record<string, number> = {};
    for (const j of jobs) counts[j.status] = (counts[j.status] ?? 0) + 1;
    return { counts, jobs: jobs.map((j) => this.jobView(j)) };
  }

  private async getJob(args: Args) {
    const id = optNumber(args, 'job_id', 1, Number.MAX_SAFE_INTEGER, true);
    if (id === undefined) fail('"job_id" is required.');
    if (!this.clientJob(id)) throw new ApiError('not_found', `No job ${id}.`);
    const wait = (optNumber(args, 'wait_seconds', 0, MAX_WAIT_SECONDS) ?? 0) * 1000;
    if (wait > 0) await Promise.race([this.deps.queue.wait(id), sleep(wait)]);
    return this.jobView(this.clientJob(id) as JobInfo);
  }

  private async cancelJob(args: Args) {
    const jobId = optNumber(args, 'job_id', 1, Number.MAX_SAFE_INTEGER, true);
    const assemblyId = optNumber(args, 'assembly_id', 1, Number.MAX_SAFE_INTEGER, true);
    const batch = optBatch(args);
    if ([jobId, assemblyId, batch].filter((v) => v !== undefined).length !== 1) {
      fail('Give exactly one of job_id, batch or assembly_id.');
    }
    if (jobId !== undefined) {
      if (!this.clientJob(jobId)) throw new ApiError('not_found', `No job ${jobId}.`);
      const cancelled = await this.deps.queue.cancel(jobId);
      return { cancelled, job: this.jobView(this.clientJob(jobId) as JobInfo) };
    }
    if (assemblyId !== undefined) {
      if (!this.deps.assemblies.get(assemblyId)) throw new ApiError('not_found', `No assembly ${assemblyId}.`);
      return { cancelled: this.deps.assemblies.cancel(assemblyId) };
    }
    return { cancelled_waiting_jobs: this.deps.queue.cancelQueued(batch, 'mcp'), note: 'Only jobs still waiting were cancelled; a running job is cancelled with its job_id.' };
  }

  // -- library -----------------------------------------------------------------------------------

  private requireItem(id: string, kinds: ItemKind[], label: string): ItemView {
    const item = this.item(id);
    if (!item) throw new ApiError('not_found', `"${label}": no library item ${id} (expected ${ITEM_ID_HINT}).`);
    if (!kinds.includes(item.kind)) fail(`"${label}": ${id} is ${item.kind === 'image' ? 'an' : 'a'} ${item.kind}, expected ${kinds.join(' or ')}.`);
    if (!fs.existsSync(item.path)) throw new ApiError('file_missing', `"${label}": the file for ${id} no longer exists (${item.path}).`);
    return item;
  }

  private listLibrary(args: Args) {
    const items = listItems(this.deps.db, {
      kind: optEnum<ItemKind>(args, 'kind', ['image', 'video', 'audio']),
      origin: optEnum<ItemOrigin>(args, 'origin', ['generated', 'imported', 'assembled']),
      batch: optBatch(args),
      limit: optNumber(args, 'limit', 1, 200, true),
      clientOnly: true,
    });
    return { count: items.length, items };
  }

  private async getItemTool(args: Args): Promise<ToolResult> {
    const id = reqString(args, 'item_id', 64);
    const item = this.item(id);
    if (!item) throw new ApiError('not_found', `No library item ${id}.`);
    const exists = fs.existsSync(item.path);
    const result: ToolResult = { data: { item, file_exists: exists } };
    if (!exists || item.kind === 'audio' || !optBool(args, 'preview', true)) return result;

    const ff = this.deps.ffmpeg();
    let poster: Buffer | null = null;
    if (ff) poster = await makePoster(ff, item.path, item.kind === 'video').catch(() => null);
    if (!poster && item.kind === 'image') poster = await this.deps.imagePreviewFallback(item.path).catch(() => null);
    if (poster) result.images = [{ mimeType: 'image/jpeg', data: poster.toString('base64') }];
    else (result.data as Record<string, unknown>).preview_note = item.kind === 'video' && !ff ? 'No preview: ffmpeg was not found.' : 'No preview could be made.';
    return result;
  }

  private async probeMediaTool(args: Args) {
    const item = this.requireItem(reqString(args, 'item_id', 64), ['image', 'video', 'audio'], 'item_id');
    const ff = this.deps.ffmpeg();
    if (!ff) {
      if (item.kind === 'image') {
        const size = this.deps.imageSize(item.path);
        return { item_id: item.id, kind: 'image', width: size?.width ?? null, height: size?.height ?? null };
      }
      throw new ApiError('ffmpeg_missing', 'ffmpeg was not found, so video and audio cannot be inspected. Set its location in KVGenius Settings.');
    }
    const info: MediaInfo = await probeMedia(ff, item.path);
    return {
      item_id: item.id,
      kind: item.kind,
      duration_seconds: info.durationSeconds,
      width: info.width,
      height: info.height,
      fps: info.fps,
      has_video: info.hasVideo,
      has_audio: info.hasAudio,
      video_codec: info.videoCodec,
      audio_codec: info.audioCodec,
    };
  }

  // -- assembling --------------------------------------------------------------------------------

  private async assembleVideo(args: Args) {
    if (!this.deps.ffmpeg()) {
      throw new ApiError('ffmpeg_missing', 'ffmpeg was not found, so videos cannot be assembled. Set its location in KVGenius Settings.');
    }
    const rawClips = args.clips;
    if (!Array.isArray(rawClips) || rawClips.length === 0) fail('"clips" must be a non-empty array.');
    if (rawClips.length > 200) fail('"clips" can hold at most 200 entries.');
    const clips = rawClips.map((entry, i) => {
      const spec = typeof entry === 'string' ? { item_id: entry } : (entry as Args | null);
      if (!spec || typeof spec !== 'object' || Array.isArray(spec)) fail(`clips[${i}] must be an item id or {item_id, seconds}.`);
      const item = this.requireItem(reqString(spec, 'item_id', 64), ['video'], `clips[${i}]`);
      const seconds = optNumber(spec, 'seconds', 0.1, 3600);
      return { itemId: item.id, path: item.path, seconds: seconds ?? null };
    });
    const audioId = optString(args, 'audio', 64);
    const audioItem = audioId ? this.requireItem(audioId, ['audio'], 'audio') : null;

    const spec: AssemblySpec = {
      clips,
      audio: audioItem ? { itemId: audioItem.id, path: audioItem.path } : null,
      transition: (optEnum<Transition>(args, 'transition', ['cut', 'crossfade']) ?? 'cut') as Transition,
      crossfadeSeconds: optNumber(args, 'crossfade_seconds', 0.05, 10) ?? 0.5,
      end: (optEnum<EndBehavior>(args, 'end', ['trim_to_video', 'trim_to_audio', 'fade_out']) ?? 'trim_to_video') as EndBehavior,
      fadeSeconds: optNumber(args, 'fade_seconds', 0.1, 30) ?? 2,
      name: optString(args, 'name', 80) ?? null,
      batch: optBatch(args) ?? null,
    };
    const started = this.deps.assemblies.start(spec);
    const wait = (optNumber(args, 'wait_seconds', 0, MAX_WAIT_SECONDS) ?? 0) * 1000;
    return wait > 0 ? await this.deps.assemblies.wait(started.id, wait) : started;
  }

  private async getAssembly(args: Args) {
    const id = optNumber(args, 'assembly_id', 1, Number.MAX_SAFE_INTEGER, true);
    if (id === undefined) fail('"assembly_id" is required.');
    if (!this.deps.assemblies.get(id)) throw new ApiError('not_found', `No assembly ${id}.`);
    const wait = (optNumber(args, 'wait_seconds', 0, MAX_WAIT_SECONDS) ?? 0) * 1000;
    return this.deps.assemblies.wait(id, wait);
  }
}
