import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { FAMILY_KIND, GenerationParams, GenerationProgress } from '../shared/types';
import { canonicalFamily, TEXT_TO_IMAGE_FAMILIES, Z_IMAGE_FAMILY } from '../shared/families';
import { I2V_FAMILY, T2V_FAMILY, videoFamilyFor } from '../shared/textToVideo';
import { V2V_FAMILY } from '../shared/videoToVideo';
import { PROFILE_FAMILIES, profileFamily, profileFamilyKey, SAMPLER_LIMITS } from '../shared/modelFamilies';
import { clampDenoise, DEFAULT_DENOISE, DENOISE_LIMITS, I2I_FAMILY, INPAINT_FAMILY, normalizeOutpaint, OUTPAINT_DEFAULT_DENOISE, OUTPAINT_FAMILY, outpaintOutputSize } from '../shared/imageToImage';
import { ModelProfile, profileSettings } from '../shared/modelProfiles';
import { findModelProfileByName, listModelProfiles } from './modelProfiles';
import { isUpscaleFamily } from '../shared/upscale';
import { JobFilter, JobInfo, JobStatus, isTerminalJobStatus } from '../shared/jobs';
import { TOOLS, ToolResult } from '../shared/tools';
import { JobQueue } from './jobQueue';
import { AssemblyManager, AssemblySpec } from './assembly';
import { FfmpegPaths, MediaInfo, EndBehavior, Transition, makePoster, probeMedia } from './mediaTools';
import { PromptStyle, combinePrompt, extraWording, styleLabel } from '../shared/styles';
import { findStyleByName, listStyles } from './styles';
import { GroupBy, importPathInWorkspaces, ItemKind, ItemOrigin, ItemView, getItem, groupItems, kindForExtension, listItems, parseItemId, upsertImport } from './library';
import { setGenerationPinned } from './db';
import { DEFAULT_WORKSPACE_ID, listWorkspaces } from './workspaces';

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
  /** The workspace the app is in. What a client makes goes there, and the styles it can name are that workspace's - unless the call names
   * another with its `workspace` argument. */
  workspaceId: () => number;
  /** The user's opt-in (Settings > Integrations) for clients to see and use hidden workspaces. Absent or false: hidden workspaces do not exist
   * as far as a client can tell - not listed, not nameable, never the default, and nothing in them is reachable by id. */
  allowHiddenWorkspaces?: () => boolean;
  /** Favorites or un-favorites a generation the way the app does (its file moves into / out of the favorites folder). Returns the file's path afterwards. */
  setFavorite: (generationId: number, favorite: boolean) => string;
}

const FAMILIES: Record<string, { kind: 'image' | 'video'; description: string }> = {
  'z-image': { kind: 'image', description: 'Fast text-to-image (Z Image Turbo).' },
  'qwen-image': { kind: 'image', description: 'Text-to-image (Qwen-Image): strong at text in the picture and at following long prompts. Slower than z-image.' },
  'flux-dev': { kind: 'image', description: 'Text-to-image (Flux.1 dev): high detail and prompt following. Slower than z-image.' },
  hidream: { kind: 'image', description: 'Text-to-image (HiDream I1 Full): a large model with four text encoders. The slowest image family.' },
  'ovis-image': { kind: 'image', description: 'Text-to-image (Ovis-Image): good at rendering text in the picture.' },
  'wan22-i2v': { kind: 'video', description: 'Image-to-video (Wan 2.2): animates a source image into a short clip at 16 fps.' },
  'wan22-t2v': { kind: 'video', description: 'Text-to-video (Wan 2.2): a short clip at 16 fps from a prompt alone.' },
};

const VIDEO_FPS = 16;
const VIDEO_LONG_SIDE = 640;
const MAX_IMPORT_FILES = 500;
const MAX_WAIT_SECONDS = 60;
const ITEM_ID_HINT = 'an item id such as "gen-12" or "imp-5"';
/** The largest picture file get_image_file will hand over (the bytes travel base64-encoded inside JSON, about a third larger). */
export const MAX_IMAGE_FILE_BYTES = 25 * 1024 * 1024;
const IMAGE_MIME_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
};
/** z-image's latent needs sides that are multiples of 16; the other families keep the coarser 64 grid until their limits are known. */
const IMAGE_SIZE_MULTIPLE_Z_IMAGE = 16;
const IMAGE_SIZE_MULTIPLE_DEFAULT = 64;

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
    return getItem(this.deps.db, id, { clientOnly: true, excludeWorkspaceIds: this.hiddenIds() });
  }

  private clientJob(id: number): JobInfo | null {
    const job = this.deps.queue.get(id);
    return job && job.source === 'mcp' && !this.hiddenIds().includes(job.workspaceId) ? job : null;
  }

  /** An assembly a client may see: not one in a hidden workspace (which looks the same as one that does not exist). */
  private assemblyVisible(id: number): boolean {
    const workspace = this.deps.assemblies.workspaceOf(id);
    return workspace !== null && !this.hiddenIds().includes(workspace);
  }

  // Hidden workspaces (a workspace property the user sets for private content) are not visible to clients unless the user opted in: they are
  // left out of every listing, cannot be named, are never the default, and their jobs, items and assemblies look like ids that do not exist.

  /** Ids of the workspaces a client must not see: the hidden ones, or none when the user allowed them. */
  private hiddenIds(): number[] {
    if (this.deps.allowHiddenWorkspaces?.() === true) return [];
    return listWorkspaces(this.deps.db, true)
      .filter((w) => w.hidden)
      .map((w) => w.id);
  }

  /** The workspaces a client may use or name, alphabetical. */
  private visibleWorkspaces() {
    const hidden = this.hiddenIds();
    return listWorkspaces(this.deps.db, true).filter((w) => !hidden.includes(w.id));
  }

  /** The workspace named by the call's optional `workspace` argument (case-insensitive), or null when it names none.
   * A hidden workspace counts as not existing (unless the user allowed clients into them), and the error does not list it. A name that matches
   * nothing is an error that lists the workspaces there are - one is never created here. */
  private namedWorkspace(args: Args): { id: number; name: string } | null {
    const wanted = optString(args, 'workspace', 200)?.trim();
    if (wanted === undefined) return null;
    const all = this.visibleWorkspaces();
    const found = all.find((w) => w.name.toLowerCase() === wanted.toLowerCase());
    if (!found) {
      const names = all.map((w) => `"${w.name}"`);
      throw new ApiError(
        'not_found',
        `No workspace named "${wanted}". ${names.length ? `Existing workspaces: ${names.join(', ')}.` : 'There are no workspaces.'} Workspaces are created in the KVGenius app (the workspace grid), never through the API - create it there first, then use its name.`
      );
    }
    return { id: found.id, name: found.name };
  }

  /** The workspace a call uses when it names none: the one the app is in - unless that one is hidden from clients, then the Default workspace
   * if it is visible, else the oldest visible one. Null when clients have no workspace at all. */
  private defaultWorkspace(): number | null {
    const current = this.deps.workspaceId();
    if (!this.hiddenIds().includes(current)) return current;
    const visible = this.visibleWorkspaces().sort((a, b) => a.id - b.id);
    return (visible.find((w) => w.id === DEFAULT_WORKSPACE_ID) ?? visible[0])?.id ?? null;
  }

  /** Where a call's work goes: the workspace it names, else the default one (see defaultWorkspace). */
  private workspaceFor(args: Args): number {
    const named = this.namedWorkspace(args);
    if (named) return named.id;
    const fallback = this.defaultWorkspace();
    if (fallback === null) {
      throw new ApiError(
        'no_workspace',
        'There is no workspace available to MCP clients. Create a workspace that is not hidden in the KVGenius app (the workspace grid), or allow clients to use hidden workspaces in Settings > Integrations.'
      );
    }
    return fallback;
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
        return { data: this.listStylesTool(args) };
      case 'list_models':
        return { data: this.listModelsTool() };
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
      case 'get_image_file':
        return this.getImageFileTool(args);
      case 'set_item_flags':
        return { data: this.setItemFlags(args) };
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
    const currentWorkspace = this.defaultWorkspace();
    const workspaces = this.visibleWorkspaces();
    const showHidden = this.deps.allowHiddenWorkspaces?.() === true;
    return {
      comfyui_reachable: comfyReachable,
      comfyui_note: comfyReachable ? null : 'ComfyUI is not reachable. Generation jobs will fail until it is running (check the address in KVGenius Settings).',
      ffmpeg_available: ff !== null,
      ffmpeg_note: ff ? null : 'ffmpeg was not found: previews of videos, probe_media and assemble_video are unavailable. Set its location in KVGenius Settings.',
      workspaces: {
        note: `Every tool that makes or lists work takes an optional \`workspace\` (a name from this list, case-insensitive). Without it, work goes to the workspace named in \`current\`. An unknown name is an error - workspaces are only created in the app. ${showHidden ? 'Hidden workspaces are included because the user allowed it in Settings.' : 'Only the workspaces listed here are available to you.'}`,
        current: workspaces.find((w) => w.id === currentWorkspace)?.name ?? null,
        available: workspaces.map((w) => (showHidden ? { name: w.name, hidden: w.hidden } : { name: w.name })),
      },
      families: [
        {
          family: 'z-image',
          ...FAMILIES['z-image'],
          tool: 'generate_image',
          fields: {
            prompt: 'text',
            style: 'optional name of a saved style (see list_styles), added after the prompt',
            width: { min: 256, max: 2048, multiple_of: IMAGE_SIZE_MULTIPLE_Z_IMAGE, default: 1024 },
            height: { min: 256, max: 2048, multiple_of: IMAGE_SIZE_MULTIPLE_Z_IMAGE, default: 1024 },
            seed: 'integer, random by default',
            workspace: 'optional workspace name (see `workspaces`); default the one the app is in',
            model: 'optional name of a saved model variant (see list_models); its steps and cfg become the defaults',
            source: 'optional library picture id: image to image - start from it instead of from nothing; width and height then default to its shape',
            strength: { min: DENOISE_LIMITS.min, max: DENOISE_LIMITS.max, default: DEFAULT_DENOISE, note: 'with source: 1 ignores the picture, small keeps most of it' },
            steps: { min: 1, max: 20, default: 8, note: 'up to 100 with a model' },
            cfg: { min: 0.5, max: 3, default: 1, note: 'up to 30 with a model' },
          },
        },
        ...TEXT_TO_IMAGE_FAMILIES.map((key) => {
          const sampler = profileFamily(key)?.sampler;
          return {
            family: key,
            ...FAMILIES[key],
            tool: 'generate_image',
            fields: {
              prompt: 'text',
              style: 'optional name of a saved style (see list_styles), added after the prompt',
              width: { min: 256, max: 2048, multiple_of: IMAGE_SIZE_MULTIPLE_DEFAULT, default: 1024 },
              height: { min: 256, max: 2048, multiple_of: IMAGE_SIZE_MULTIPLE_DEFAULT, default: 1024 },
              seed: 'integer, random by default',
              workspace: 'optional workspace name (see `workspaces`); default the one the app is in',
              model: 'optional name of a saved model variant of this family (see list_models); its steps and cfg become the defaults',
              steps: { min: SAMPLER_LIMITS.steps.min, max: SAMPLER_LIMITS.steps.max, default: sampler?.steps },
              cfg: { min: SAMPLER_LIMITS.cfg.min, max: SAMPLER_LIMITS.cfg.max, default: sampler?.cfg, ...(key === 'flux-dev' ? { note: 'the guidance value of the model (its sampler CFG stays 1)' } : {}) },
              source: 'not available: this family has text to image only',
            },
          };
        }),
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
            workspace: 'optional workspace name (see `workspaces`); default the one the app is in',
          },
        },
        {
          family: 'wan22-t2v',
          ...FAMILIES['wan22-t2v'],
          tool: 'generate_video',
          fields: {
            prompt: 'text describing the scene and the motion',
            source: 'leave out - passing a source picks image-to-video instead',
            seconds: { min: 1, max: 12, default: 5, note: 'a clip is ~5s; longer takes proportionally longer' },
            width: { min: 256, max: 1280, multiple_of: 16, default: 640 },
            height: { min: 256, max: 1280, multiple_of: 16, default: 640 },
            seed: 'integer, random by default',
            workspace: 'optional workspace name (see `workspaces`); default the one the app is in',
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
    const workspaceId = this.workspaceFor(args);

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
    const hidden = this.hiddenIds();
    for (const file of files.slice(0, MAX_IMPORT_FILES)) {
      // A path already imported into a hidden workspace belongs to it (an import's path is unique), and it is not the client's to see or regroup.
      if (importPathInWorkspaces(this.deps.db, file, hidden)) continue;
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
      items.push(upsertImport(this.deps.db, { path: file, kind, origin: 'imported', width, height, duration, batch, workspaceId }));
    }
    return { imported: items.length, truncated, limit: truncated ? MAX_IMPORT_FILES : undefined, items };
  }

  // -- generating --------------------------------------------------------------------------------

  private generateImage(args: Args) {
    const prompt = reqString(args, 'prompt');
    const workspaceId = this.workspaceFor(args);
    // With no family given, a saved model decides it (the user named a Flux model: that is the Flux family); otherwise Z-Image.
    const named = optString(args, 'model', 200);
    const namedProfile = named ? findModelProfileByName(this.deps.db, named.trim()) : null;
    const family = canonicalFamily(optString(args, 'family', 64) ?? (namedProfile && FAMILY_KIND[namedProfile.family] === 'image' ? namedProfile.family : Z_IMAGE_FAMILY));
    if (family === INPAINT_FAMILY) fail('Inpainting is not available here yet - it needs a painted mask, which only the app can make.');
    if (family === OUTPAINT_FAMILY) fail('Outpainting is not a family of its own here: use family "z-image" and pass `source` (a library picture) with `extend` (pixels to add on each side).');
    if (family === I2I_FAMILY) fail('Image to image is not a family of its own here: use family "z-image" and pass `source` (a library picture to start from).');
    if (FAMILY_KIND[family] !== 'image') fail(`"${family}" is not an image family. Image families: ${Object.keys(FAMILY_KIND).filter((f) => FAMILY_KIND[f] === 'image').join(', ')}.`);
    // A style's words are added here, so the job (and the Library record) holds the full prompt that is
    // sent. With no style the prompt is passed through untouched.
    const { style, elements } = this.resolveWording(args, workspaceId);
    // With a `source` picture this is image to image: same model, but the sampler starts from that picture.
    const sourceId = optString(args, 'source', 64);
    if ((sourceId !== undefined || (args.extend !== undefined && args.extend !== null)) && family !== Z_IMAGE_FAMILY) {
      fail(`"${family}" does text to image only: \`source\` and \`extend\` (image to image, outpainting) are available with family "z-image".`);
    }
    const source = sourceId === undefined ? null : this.requireItem(sourceId, ['image'], 'source');
    // A saved model swaps the files and sampler inside the family's graph; it is resolved here, so the job and
    // the Library record carry exactly what runs (the profile may change or go away later).
    const model = this.resolveModel(optString(args, 'model', 200), family);
    // A family other than Z-Image has no "built-in" limits of its own: the built-in run uses the family's own steps and CFG.
    const samplerDefaults = model?.sampler ?? (TEXT_TO_IMAGE_FAMILIES.includes(family) ? profileFamily(family)?.sampler : null) ?? null;
    // Without a width or height of its own, the picture's shape decides the size (long side 1024), like Generate does.
    const sourceSize = source ? (source.width && source.height ? { width: source.width, height: source.height } : this.deps.imageSize(source.path)) : null;
    const fit = sourceSize ? 1024 / Math.max(sourceSize.width, sourceSize.height) : 1;
    const defaultWidth = sourceSize ? Math.round(sourceSize.width * fit) : 1024;
    const defaultHeight = sourceSize ? Math.round(sourceSize.height * fit) : 1024;
    // `extend` makes it outpainting: the picture grows by those pixels per side and only the new area is drawn.
    const extend = args.extend === undefined || args.extend === null ? null : normalizeOutpaint(args.extend);
    if (args.extend !== undefined && args.extend !== null && !extend) fail('"extend" needs at least one of left, top, right, bottom above 0 (pixels to add on that side).');
    if (extend && !source) fail('"extend" needs a "source": the library picture to extend.');
    if (extend && (!sourceSize || sourceSize.width <= 0)) throw new ApiError('unreadable_image', `Could not read the size of ${source?.id}, so it cannot be extended.`);
    const extendedSize = extend && source && sourceSize ? outpaintOutputSize(source.width || sourceSize.width, source.height || sourceSize.height, extend) : null;
    const multiple = family === Z_IMAGE_FAMILY ? IMAGE_SIZE_MULTIPLE_Z_IMAGE : IMAGE_SIZE_MULTIPLE_DEFAULT;
    const params: GenerationParams = {
      prompt: combinePrompt(prompt, extraWording(elements, style)),
      ...(styleLabel(elements, style) ? { styleName: styleLabel(elements, style) as string } : {}),
      width: extendedSize ? extendedSize.width : snap(optNumber(args, 'width', 64, 8192, true) ?? defaultWidth, multiple, 256, 2048),
      height: extendedSize ? extendedSize.height : snap(optNumber(args, 'height', 64, 8192, true) ?? defaultHeight, multiple, 256, 2048),
      seed: optNumber(args, 'seed', 0, 2 ** 32 - 1, true) ?? Math.floor(Math.random() * 2 ** 32),
      steps: samplerDefaults ? (optNumber(args, 'steps', SAMPLER_LIMITS.steps.min, SAMPLER_LIMITS.steps.max, true) ?? samplerDefaults.steps) : (optNumber(args, 'steps', 1, 20, true) ?? 8),
      cfg: samplerDefaults ? (optNumber(args, 'cfg', SAMPLER_LIMITS.cfg.min, SAMPLER_LIMITS.cfg.max) ?? samplerDefaults.cfg) : (optNumber(args, 'cfg', 0.5, 3) ?? 1),
      ...(model ? { modelName: model.name, modelSettings: profileSettings(model) } : {}),
      ...(source && extend ? { sourceImagePath: source.path, outpaint: extend, denoise: clampDenoise(optNumber(args, 'strength', DENOISE_LIMITS.min, DENOISE_LIMITS.max), OUTPAINT_DEFAULT_DENOISE) } : {}),
      ...(source && !extend ? { sourceImagePath: source.path, denoise: clampDenoise(optNumber(args, 'strength', DENOISE_LIMITS.min, DENOISE_LIMITS.max)) } : {}),
    };
    return this.jobView(this.deps.queue.submit({ family: extend ? OUTPAINT_FAMILY : source ? I2I_FAMILY : family, params, source: 'mcp', batch: optBatch(args), workspaceId }));
  }

  /** The saved model named `name`, or null for the built-in one (no name, "default", or its own name). */
  private resolveModel(name: string | undefined, family: string): ModelProfile | null {
    const wanted = name?.trim();
    const def = profileFamily(profileFamilyKey(family));
    if (!wanted || wanted.toLowerCase() === 'default' || wanted.toLowerCase() === def?.builtInName.toLowerCase()) return null;
    const profile = findModelProfileByName(this.deps.db, wanted);
    if (!profile) {
      const names = listModelProfiles(this.deps.db).map((p) => `"${p.name}"`);
      throw new ApiError(
        'not_found',
        `No model named "${wanted}". ${names.length ? `Saved models: ${names.join(', ')}.` : 'There are no saved models yet - they are added in the KVGenius Models page.'} Omit \`model\` for the built-in one.`
      );
    }
    if (profile.family !== profileFamilyKey(family)) fail(`The model "${profile.name}" is for the "${profile.family}" family, not "${family}".`);
    return profile;
  }

  private listModelsTool() {
    const builtIns = PROFILE_FAMILIES.map((def) => ({
      name: def.builtInName,
      family: def.family,
      tool: FAMILY_KIND[def.family] === 'video' ? 'generate_video' : 'generate_image',
      built_in: true,
      ...(def.sampler ? { steps: def.sampler.steps, cfg: def.sampler.cfg } : {}),
    }));
    const saved = listModelProfiles(this.deps.db).map((p) => ({
      name: p.name,
      family: p.family,
      tool: FAMILY_KIND[p.family] === 'video' ? 'generate_video' : 'generate_image',
      ...(profileFamily(p.family)?.sampler ? { steps: p.sampler.steps, cfg: p.sampler.cfg, sampler: p.sampler.sampler, scheduler: p.sampler.scheduler } : {}),
      files: p.files,
    }));
    return {
      models: [...builtIns, ...saved],
      note: saved.length
        ? 'Pass a name as `model` to the tool listed for it. For images its steps and cfg are the defaults unless you pass your own; video quality is decided by KVGenius.'
        : 'Only the built-in models so far - the user adds others in the KVGenius Models page.',
    };
  }

  /** The saved style (`style`, a name) and elements (`elements`, a list of names) a generate_image call asks for. Throws, naming what exists, for a
   * name that is not saved or is the wrong kind - a style is a look (one), an element is a reusable part of the picture (any number). */
  private resolveWording(args: Args, workspaceId: number): { style: PromptStyle | null; elements: PromptStyle[] } {
    const all = listStyles(this.deps.db, workspaceId);
    const names = (kind: 'style' | 'element') => all.filter((s) => s.kind === kind).map((s) => `"${s.name}"`);
    const find = (name: string, kind: 'style' | 'element', argument: string): PromptStyle => {
      const found = findStyleByName(this.deps.db, name, workspaceId);
      if (!found) {
        const saved = names(kind);
        throw new ApiError(
          'not_found',
          `No ${kind} named "${name.trim()}". ${saved.length ? `Saved ${kind === 'style' ? 'styles' : 'elements'}: ${saved.join(', ')}.` : `There are none saved yet - they are created in the KVGenius Styles tab.`}`
        );
      }
      if (found.kind !== kind) fail(`"${found.name}" is ${found.kind === 'style' ? 'a style' : 'an element'}: pass it as \`${found.kind === 'style' ? 'style' : 'elements'}\`, not \`${argument}\`.`);
      return found;
    };
    const styleName = optString(args, 'style', 200);
    const style = styleName === undefined ? null : find(styleName, 'style', 'style');
    const raw = args.elements;
    if (raw !== undefined && raw !== null && (!Array.isArray(raw) || raw.some((n) => typeof n !== 'string' || n.trim() === ''))) fail('"elements" must be a list of saved element names.');
    const elements: PromptStyle[] = [];
    for (const name of (raw as string[] | undefined) ?? []) {
      const element = find(name, 'element', 'elements');
      if (!elements.some((e) => e.id === element.id)) elements.push(element);
    }
    return { style, elements };
  }

  private listStylesTool(args: Args) {
    const styles = listStyles(this.deps.db, this.workspaceFor(args)).map((s) => ({ name: s.name, kind: s.kind, text: s.text }));
    return {
      styles,
      note: styles.length
        ? 'A "style" is a look (anime, oil painting): pass its name as `style` to generate_image (one per picture). An "element" is a reusable part of the picture (an outfit, a character): pass any number of names as `elements`. Elements are added after your prompt, then the style, each after a comma.'
        : 'Nothing saved yet - the user creates styles and elements in the KVGenius Styles tab.',
    };
  }

  private generateVideo(args: Args) {
    const prompt = reqString(args, 'prompt');
    const workspaceId = this.workspaceFor(args);
    // With a source picture it is image to video, without one text to video - unless the family says otherwise.
    const sourceId = optString(args, 'source', 64);
    const family = optString(args, 'family', 64) ?? videoFamilyFor(sourceId !== undefined);
    if (FAMILY_KIND[family] !== 'video' || isUpscaleFamily(family)) fail(`"${family}" is not a video family. Video families: ${Object.keys(FAMILY_KIND).filter((f) => FAMILY_KIND[f] === 'video' && !isUpscaleFamily(f)).join(', ')}.`);
    if (family === V2V_FAMILY) fail('Video to video is only available in the app.');
    if (family === I2V_FAMILY && sourceId === undefined) fail('"source" is required for image-to-video (pass a library image id, or use family "wan22-t2v" for text-to-video).');
    if (family === T2V_FAMILY && sourceId !== undefined) fail('Text-to-video takes no "source". Leave it out, or use family "wan22-i2v" to animate that picture.');
    const source = sourceId === undefined ? null : this.requireItem(sourceId, ['image'], 'source');

    const size = source ? (source.width && source.height ? { width: source.width, height: source.height } : this.deps.imageSize(source.path)) : { width: VIDEO_LONG_SIDE, height: VIDEO_LONG_SIDE };
    let width = optNumber(args, 'width', 16, 8192, true);
    let height = optNumber(args, 'height', 16, 8192, true);
    if (width === undefined || height === undefined) {
      if (!size) throw new ApiError('unreadable_image', `Could not read the size of ${source?.id}; pass width and height explicitly.`);
      // Same rule as the Generate page: keep the aspect ratio at ~640 on the long side.
      const scale = VIDEO_LONG_SIDE / Math.max(size.width, size.height);
      width = width ?? Math.max(256, Math.round((size.width * scale) / 16) * 16);
      height = height ?? Math.max(256, Math.round((size.height * scale) / 16) * 16);
    }
    const seconds = optNumber(args, 'seconds', 1, 12) ?? 5;
    // A saved video model swaps the Wan files inside the same graph, resolved here like generate_image's.
    const model = this.resolveModel(optString(args, 'model', 200), family);
    const params: GenerationParams = {
      prompt,
      width: snap(width, 16, 256, 1280),
      height: snap(height, 16, 256, 1280),
      seed: optNumber(args, 'seed', 0, 2 ** 32 - 1, true) ?? Math.floor(Math.random() * 2 ** 32),
      // The video template ignores these; they match the Generate page's defaults so timing history stays comparable.
      steps: 8,
      cfg: 1,
      length: 4 * Math.round(seconds * (VIDEO_FPS / 4)) + 1,
      ...(source ? { sourceImagePath: source.path } : {}),
      ...(model ? { modelName: model.name, modelSettings: profileSettings(model) } : {}),
    };
    return this.jobView(this.deps.queue.submit({ family, params, source: 'mcp', batch: optBatch(args), workspaceId }));
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
      // Only narrowed when a workspace is named: with none, jobs from every workspace are listed - except hidden ones (see hiddenIds).
      workspaceId: this.namedWorkspace(args)?.id,
      excludeWorkspaceIds: this.hiddenIds(),
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
      if (!this.assemblyVisible(assemblyId)) throw new ApiError('not_found', `No assembly ${assemblyId}.`);
      return { cancelled: this.deps.assemblies.cancel(assemblyId) };
    }
    return { cancelled_waiting_jobs: this.deps.queue.cancelQueued(batch, 'mcp', this.hiddenIds()), note: 'Only jobs still waiting were cancelled; a running job is cancelled with its job_id.' };
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
    const filter = {
      kind: optEnum<ItemKind>(args, 'kind', ['image', 'video', 'audio']),
      origin: optEnum<ItemOrigin>(args, 'origin', ['generated', 'imported', 'assembled']),
      batch: optBatch(args),
      limit: optNumber(args, 'limit', 1, 200, true),
      clientOnly: true,
      // Only narrowed when a workspace is named: with none, items from every workspace are listed - except hidden ones (see hiddenIds).
      workspaceId: this.namedWorkspace(args)?.id,
      excludeWorkspaceIds: this.hiddenIds(),
      favorite: args.favorite === undefined || args.favorite === null ? undefined : optBool(args, 'favorite', false),
      pinned: args.pinned === undefined || args.pinned === null ? undefined : optBool(args, 'pinned', false),
    };
    const by = optEnum<GroupBy>(args, 'group_by', ['prompt', 'batch']);
    if (by) {
      const { groups, truncated } = groupItems(this.deps.db, filter, by);
      return {
        group_by: by,
        count: groups.length,
        truncated,
        note: truncated ? 'Only the newest 5000 matching items were grouped; narrow with batch, workspace or the favorite / pinned filters.' : undefined,
        groups: groups.map((g) => ({
          [by]: g.key,
          count: g.count,
          favorites: g.favorites,
          pinned: g.pinned,
          cover: g.cover,
          item_ids: g.itemIds,
          item_ids_truncated: g.count > g.itemIds.length,
        })),
      };
    }
    const items = listItems(this.deps.db, filter);
    return { count: items.length, items };
  }

  /** Favorites / pins generated items, several at a time. All ids are checked first, so a bad one changes nothing. */
  private setItemFlags(args: Args) {
    const raw = args.item_ids;
    if (!Array.isArray(raw) || raw.length === 0 || raw.some((id) => typeof id !== 'string' || id.trim() === '')) fail('"item_ids" must be a non-empty list of item ids.');
    if (raw.length > 200) fail('"item_ids" can hold at most 200 ids.');
    const favorite = args.favorite === undefined || args.favorite === null ? undefined : optBool(args, 'favorite', false);
    const pinned = args.pinned === undefined || args.pinned === null ? undefined : optBool(args, 'pinned', false);
    if (favorite === undefined && pinned === undefined) fail('Give `favorite` and/or `pinned` (true or false).');
    const ids = [...new Set((raw as string[]).map((id) => id.trim()))];
    const generations = ids.map((id) => {
      const item = this.item(id);
      if (!item) throw new ApiError('not_found', `No library item ${id}.`);
      const ref = parseItemId(id);
      if (!ref || ref.table !== 'gen') fail(`${id} is an imported or assembled file: only generated items (ids like "gen-12") can be favorited or pinned.`);
      return { item, generationId: ref.n };
    });
    for (const { generationId } of generations) {
      if (favorite !== undefined) this.deps.setFavorite(generationId, favorite);
      if (pinned !== undefined) setGenerationPinned(this.deps.db, generationId, pinned);
    }
    const items = generations.map(({ item }) => this.item(item.id) as ItemView);
    return {
      updated: items.length,
      items,
      note: favorite === undefined ? undefined : 'Favoriting moves a picture\'s file into the favorites folder (and un-favoriting moves it back): use the `path` in these results, not an earlier one.',
    };
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

  /** The original picture file, byte for byte, base64-encoded in `images[0]` (the same slot get_item's preview uses, but with the file's own
   * mimeType). Read-only: nothing is ever written anywhere. Out-of-scope and missing ids look the same as in get_item. */
  private async getImageFileTool(args: Args): Promise<ToolResult> {
    const item = this.requireItem(reqString(args, 'item_id', 64), ['image'], 'item_id');
    const mimeType = IMAGE_MIME_TYPES[path.extname(item.path).toLowerCase()];
    if (!mimeType) fail(`"item_id": ${item.id} (${item.name}) is not a picture file type this tool can return.`);
    const size = fs.statSync(item.path).size;
    if (size > MAX_IMAGE_FILE_BYTES) {
      throw new ApiError('too_large', `${item.id} is ${(size / 1024 / 1024).toFixed(1)} MB; get_image_file returns files up to ${MAX_IMAGE_FILE_BYTES / 1024 / 1024} MB.`);
    }
    const bytes = await fs.promises.readFile(item.path);
    return {
      data: { item_id: item.id, file_name: item.name, mime_type: mimeType, bytes: bytes.length, width: item.width, height: item.height, note: 'The file itself is in images[0].data, base64-encoded.' },
      images: [{ mimeType, data: bytes.toString('base64') }],
    };
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
    const workspaceId = this.workspaceFor(args);
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
      workspaceId,
    };
    const started = this.deps.assemblies.start(spec);
    const wait = (optNumber(args, 'wait_seconds', 0, MAX_WAIT_SECONDS) ?? 0) * 1000;
    return wait > 0 ? await this.deps.assemblies.wait(started.id, wait) : started;
  }

  private async getAssembly(args: Args) {
    const id = optNumber(args, 'assembly_id', 1, Number.MAX_SAFE_INTEGER, true);
    if (id === undefined) fail('"assembly_id" is required.');
    if (!this.assemblyVisible(id)) throw new ApiError('not_found', `No assembly ${id}.`);
    const wait = (optNumber(args, 'wait_seconds', 0, MAX_WAIT_SECONDS) ?? 0) * 1000;
    return this.deps.assemblies.wait(id, wait);
  }
}
