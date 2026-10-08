import { randomUUID } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { GpuInfo, parseSystemStats } from '../shared/gpuInfo';
import { GenerationParams, GenerationProgress } from '../shared/types';
import { videoQualityFromCfg } from '../shared/videoQuality';
import { ComfyMessage, ProgressTracker, RunTimings } from './progressTracker';
import { getEffectiveComfyUIHost } from './dbLocation';
import zImageTurboTemplate from './templates/z-image.json';
import wan22I2vTemplate from './templates/wan22-i2v.json';
import wan22T2vTemplate from './templates/wan22-t2v.json';
import zImageI2iTemplate from './templates/z-image-i2i.json';
import zImageInpaintTemplate from './templates/z-image-inpaint.json';
import upscaleImageTemplate from './templates/upscale-image.json';
import upscaleVideoTemplate from './templates/upscale-video.json';
import { UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY } from '../shared/upscale';
import { canonicalFamily } from '../shared/families';
import { I2I_FAMILY, INPAINT_FAMILY } from '../shared/imageToImage';
import { I2V_FAMILY, T2V_FAMILY } from '../shared/textToVideo';
import { fillImageToImage, fillInpaint } from './imageToImagePatch';
import { applyModelSettings } from './modelPatch';
import { FOLDER_LOADERS, MODEL_FOLDERS } from '../shared/modelManifest';
import { emptyInstalled, InstalledModels, parseChoiceList } from '../shared/modelStatus';

// Imported directly (not read from disk at runtime via fs) so tsc inlines the JSON into the
// compiled output - `tsc -p tsconfig.main.json` only compiles .ts files, it doesn't copy
// arbitrary assets into dist/, so a fs.readFileSync(path.join(__dirname, ...)) here would
// silently work in dev (where src/ and dist/ can end up looking similar) and fail in a real
// build with ENOENT once the template stops existing next to the compiled .js. One entry per
// model family template; add to this map as more templates are added.
const TEMPLATES: Record<string, Record<string, unknown>> = {
  'z-image': zImageTurboTemplate,
  [I2I_FAMILY]: zImageI2iTemplate,
  [INPAINT_FAMILY]: zImageInpaintTemplate,
  'wan22-i2v': wan22I2vTemplate,
  [T2V_FAMILY]: wan22T2vTemplate,
  [UPSCALE_FAMILY]: upscaleImageTemplate,
  [UPSCALE_VIDEO_FAMILY]: upscaleVideoTemplate,
};

// ComfyUI Desktop (the Electron distribution this app targets) defaults to port 8000, not
// the classic standalone ComfyUI server's 8188 - different implementations, different defaults.
export const DEFAULT_COMFYUI_HOST = 'http://localhost:8000';

/**
 * Node IDs in src/main/templates/z-image.json that patchTemplate() fills in.
 * If that file changes, keep these in sync - see its README-equivalent comment
 * block at the top of comfyui-templates.md (to be written once a second
 * template exists and this needs a real per-family map).
 */
const Z_IMAGE_TURBO_NODE_MAP = {
  prompt: '57:27',
  sampler: '57:3',
  latent: '57:13',
};

/**
 * Node IDs in src/main/templates/wan22-i2v.json. This is a 2-stage (high-noise then
 * low-noise) KSamplerAdvanced pipeline gated by a "4-step LoRA" switch chain (node 129:131)
 * that the template ships already enabled. Only that one boolean (fastLoraSwitch) is patched, to
 * offer a Fast/High quality choice - the steps, CFG and model choices it selects between stay
 * exactly as the template author set them (see CLAUDE.md's curated-field philosophy).
 * Only samplerHighNoise's seed is patched - samplerLowNoise (129:85) has add_noise:'disable'
 * and return_with_leftover_noise from stage 1, so its own noise_seed field is inert.
 */
const WAN22_I2V_NODE_MAP = {
  loadImage: '97',
  positivePrompt: '129:93',
  imageToVideo: '129:98',
  samplerHighNoise: '129:86',
  fastLoraSwitch: '129:131',
};

/** Node IDs in src/main/templates/upscale-image.json: load image -> model upscale -> resize to the
 * exact requested size -> save. Only core ComfyUI nodes, no custom packs. */
const UPSCALE_NODE_MAP = {
  loadImage: '1',
  modelLoader: '2',
  resize: '4',
};

/** Node IDs in src/main/templates/upscale-video.json: load video -> split into frames -> model upscale
 * -> resize to the exact size -> reassemble (keeping the audio and frame rate) -> save. Core nodes only. */
const UPSCALE_VIDEO_NODE_MAP = {
  loadVideo: '1',
  modelLoader: '2',
  resize: '5',
};

export class ComfyUIUnavailableError extends Error {}
export class GenerationCancelledError extends Error {}

async function comfyRequest(path: string, init?: RequestInit): Promise<Response> {
  const host = getEffectiveComfyUIHost();
  const url = `${host}${path}`;
  let resp: Response;
  try {
    resp = await fetch(url, init);
  } catch (err) {
    if (init?.signal?.aborted) throw new GenerationCancelledError('Generation cancelled.');
    throw new ComfyUIUnavailableError(`ComfyUI not reachable at ${host}: ${String(err)}`);
  }
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new ComfyUIUnavailableError(`ComfyUI returned ${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp;
}

// Tracks the one generation this app can have in flight at a time, so cancelCurrentGeneration()
// can (a) actually tell ComfyUI to stop the work - not just give up waiting for it client-side,
// which was a real bug: the client hitting its own timeout left the job running on the GPU with
// nothing left listening for the result - and (b) abort the local fetch/poll loop so the
// blocked IPC call returns promptly instead of hanging until whatever timeout is set below.
let currentPromptId: string | null = null;
let currentAbortController: AbortController | null = null;

/** Best-effort: asks ComfyUI to interrupt whatever it's currently running and removes the
 * tracked prompt from its queue if it hadn't started yet, then aborts the local wait. Safe to
 * call with nothing in flight (no-ops). Does not itself throw - the in-flight generate() call
 * surfaces the actual cancellation via GenerationCancelledError once its fetch/poll aborts. */
export async function cancelCurrentGeneration(): Promise<void> {
  const promptId = currentPromptId;
  if (promptId) {
    try {
      await comfyRequest('/interrupt', { method: 'POST' });
    } catch {
      // ComfyUI may already be unreachable/gone - the local abort below still unblocks the UI.
    }
    try {
      await comfyRequest('/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delete: [promptId] }),
      });
    } catch {
      // Best-effort - it may already have started (not in queue) or already finished.
    }
  }
  currentAbortController?.abort();
}

/** The files one of ComfyUI's loader nodes offers, read from the node's own list of choices. */
async function listLoaderChoices(node: string, input: string): Promise<string[]> {
  const resp = await comfyRequest(`/object_info/${node}`, { signal: AbortSignal.timeout(10000) });
  const data = (await resp.json()) as Record<string, { input?: { required?: Record<string, unknown> } } | undefined>;
  return parseChoiceList(data[node]?.input?.required?.[input]);
}

/** Upscale models ComfyUI can load. */
export function listUpscaleModels(): Promise<string[]> {
  return listLoaderChoices(FOLDER_LOADERS.upscale_models.node, FOLDER_LOADERS.upscale_models.input);
}

/** The sampler and scheduler names this ComfyUI offers (KSampler's own choice lists). */
export async function listSamplerChoices(): Promise<{ samplers: string[]; schedulers: string[] }> {
  const [samplers, schedulers] = await Promise.all([listLoaderChoices('KSampler', 'sampler_name'), listLoaderChoices('KSampler', 'scheduler')]);
  return { samplers, schedulers };
}

/** Every model file ComfyUI can see in the folders the app cares about. A loader this ComfyUI does not
 * know is treated as having no files; if none of them can be asked at all, ComfyUI is unreachable. */
export async function listInstalledModels(): Promise<InstalledModels> {
  const installed = emptyInstalled();
  const results = await Promise.allSettled(
    MODEL_FOLDERS.map(async (folder) => {
      installed[folder] = await listLoaderChoices(FOLDER_LOADERS[folder].node, FOLDER_LOADERS[folder].input);
    }),
  );
  const failures = results.filter((r): r is PromiseRejectedResult => r.status === 'rejected');
  if (failures.length === results.length) throw failures[0].reason;
  return installed;
}

/** The compute devices ComfyUI is using (name and memory), or none if it cannot be asked. This is the card that
 * actually runs generations - which is also right for a ComfyUI on another machine. */
export async function getGpuInfo(): Promise<GpuInfo[]> {
  try {
    const resp = await comfyRequest('/system_stats', { signal: AbortSignal.timeout(5000) });
    return parseSystemStats(await resp.json());
  } catch {
    return [];
  }
}

export async function isAvailable(): Promise<boolean> {
  try {
    await comfyRequest('/system_stats', { signal: AbortSignal.timeout(5000) });
    return true;
  } catch {
    return false;
  }
}

function loadTemplate(family: string): Record<string, unknown> {
  const template = TEMPLATES[canonicalFamily(family)];
  if (!template) {
    throw new Error(`No ComfyUI workflow template registered for family '${family}'`);
  }
  return template;
}

/** Uploads a local image file to ComfyUI's input directory so a LoadImage node can
 * reference it by filename. Returns the filename ComfyUI stored it under. */
async function uploadSourceImage(filePath: string, signal: AbortSignal): Promise<string> {
  const bytes = fs.readFileSync(filePath);
  const form = new FormData();
  form.append('image', new Blob([bytes]), path.basename(filePath));
  const resp = await comfyRequest('/upload/image', { method: 'POST', body: form, signal });
  const data = (await resp.json()) as { name?: string };
  if (!data.name) {
    throw new ComfyUIUnavailableError(`ComfyUI did not return a filename for the uploaded image: ${JSON.stringify(data)}`);
  }
  return data.name;
}

async function patchTemplate(
  family: string,
  template: Record<string, unknown>,
  params: GenerationParams,
  signal: AbortSignal
): Promise<Record<string, unknown>> {
  const workflow = JSON.parse(JSON.stringify(template));

  if (family === UPSCALE_FAMILY) {
    if (!params.sourceImagePath) throw new Error('Upscaling needs a source image.');
    if (!params.upscaleModel) throw new Error('Choose an upscale model first.');
    const uploadedName = await uploadSourceImage(params.sourceImagePath, signal);
    (workflow[UPSCALE_NODE_MAP.loadImage] as { inputs: Record<string, unknown> }).inputs.image = uploadedName;
    (workflow[UPSCALE_NODE_MAP.modelLoader] as { inputs: Record<string, unknown> }).inputs.model_name = params.upscaleModel;
    const resize = workflow[UPSCALE_NODE_MAP.resize] as { inputs: Record<string, unknown> };
    resize.inputs.width = params.width;
    resize.inputs.height = params.height;
    return workflow;
  }

  if (family === UPSCALE_VIDEO_FAMILY) {
    if (!params.sourceVideoPath) throw new Error('Upscaling needs a source video.');
    if (!params.upscaleModel) throw new Error('Choose an upscale model first.');
    // ComfyUI's upload endpoint stores any file in its input folder; LoadVideo then finds it by name.
    const uploadedName = await uploadSourceImage(params.sourceVideoPath, signal);
    (workflow[UPSCALE_VIDEO_NODE_MAP.loadVideo] as { inputs: Record<string, unknown> }).inputs.file = uploadedName;
    (workflow[UPSCALE_VIDEO_NODE_MAP.modelLoader] as { inputs: Record<string, unknown> }).inputs.model_name = params.upscaleModel;
    const resize = workflow[UPSCALE_VIDEO_NODE_MAP.resize] as { inputs: Record<string, unknown> };
    resize.inputs.width = params.width;
    resize.inputs.height = params.height;
    return workflow;
  }

  if (canonicalFamily(family) === INPAINT_FAMILY) {
    if (!params.sourceImagePath) throw new Error('Inpainting needs a source image.');
    if (!params.maskImagePath) throw new Error('Inpainting needs a mask - paint the spots to change first.');
    const imageName = await uploadSourceImage(params.sourceImagePath, signal);
    const maskName = await uploadSourceImage(params.maskImagePath, signal);
    fillInpaint(workflow, params, imageName, maskName);
    return workflow;
  }

  if (canonicalFamily(family) === I2I_FAMILY) {
    if (!params.sourceImagePath) throw new Error('Image to image needs a start picture.');
    const uploadedName = await uploadSourceImage(params.sourceImagePath, signal);
    fillImageToImage(workflow, params, uploadedName);
    return workflow;
  }

  if (family === I2V_FAMILY || family === T2V_FAMILY) {
    // Text to video is the same graph with the start picture replaced by an empty latent (same node ids).
    if (family === I2V_FAMILY) {
      if (!params.sourceImagePath) {
        throw new Error('Image to video requires a source image.');
      }
      const uploadedName = await uploadSourceImage(params.sourceImagePath, signal);
      const loadImageNode = workflow[WAN22_I2V_NODE_MAP.loadImage] as { inputs: Record<string, unknown> };
      loadImageNode.inputs.image = uploadedName;
    }

    const promptNode = workflow[WAN22_I2V_NODE_MAP.positivePrompt] as { inputs: Record<string, unknown> };
    promptNode.inputs.text = params.prompt;

    const imageToVideoNode = workflow[WAN22_I2V_NODE_MAP.imageToVideo] as { inputs: Record<string, unknown> };
    imageToVideoNode.inputs.width = params.width;
    imageToVideoNode.inputs.height = params.height;
    imageToVideoNode.inputs.length = params.length ?? 81;

    const samplerNode = workflow[WAN22_I2V_NODE_MAP.samplerHighNoise] as { inputs: Record<string, unknown> };
    samplerNode.inputs.noise_seed = params.seed;

    // One boolean flips the whole switch chain between the 4-step LoRA path and the 20-step path.
    const loraSwitchNode = workflow[WAN22_I2V_NODE_MAP.fastLoraSwitch] as { inputs: Record<string, unknown> };
    loraSwitchNode.inputs.value = videoQualityFromCfg(params.cfg) === 'fast';

    // A model profile swaps the model, text encoder, VAE and LoRA files inside this same graph.
    if (params.modelSettings) applyModelSettings(workflow, canonicalFamily(family), params.modelSettings);

    return workflow;
  }

  const promptNode = workflow[Z_IMAGE_TURBO_NODE_MAP.prompt] as { inputs: Record<string, unknown> };
  promptNode.inputs.text = params.prompt;

  const samplerNode = workflow[Z_IMAGE_TURBO_NODE_MAP.sampler] as { inputs: Record<string, unknown> };
  samplerNode.inputs.seed = params.seed;
  samplerNode.inputs.steps = params.steps;
  samplerNode.inputs.cfg = params.cfg;

  const latentNode = workflow[Z_IMAGE_TURBO_NODE_MAP.latent] as { inputs: Record<string, unknown> };
  latentNode.inputs.width = params.width;
  latentNode.inputs.height = params.height;

  // A model profile swaps the loader files and sampler values inside this same graph.
  if (params.modelSettings) applyModelSettings(workflow, canonicalFamily(family), params.modelSettings);

  return workflow;
}

async function submit(workflow: Record<string, unknown>, signal: AbortSignal, clientId: string): Promise<string> {
  const resp = await comfyRequest('/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: workflow, client_id: clientId }),
    signal,
  });
  const data = (await resp.json()) as { prompt_id?: string };
  if (!data.prompt_id) {
    throw new ComfyUIUnavailableError(`ComfyUI did not return a prompt_id: ${JSON.stringify(data)}`);
  }
  return data.prompt_id;
}

interface HistoryFile {
  filename: string;
  subfolder: string;
  type: string;
}

interface HistoryEntry {
  outputs?: Record<string, Record<string, HistoryFile[]>>;
  /** ComfyUI marks a run that failed here, with the reason in its messages. */
  status?: { status_str?: string; messages?: unknown[] };
}

/** The reason ComfyUI gave for a failed run (its `execution_error` message), if there is one. */
export function describeExecutionError(entry: HistoryEntry): string | null {
  if (entry.status?.status_str !== 'error') return null;
  for (const message of entry.status.messages ?? []) {
    if (!Array.isArray(message) || message[0] !== 'execution_error') continue;
    const data = (message[1] ?? {}) as { exception_message?: unknown; node_type?: unknown };
    const text = typeof data.exception_message === 'string' ? data.exception_message.trim() : '';
    if (text) return typeof data.node_type === 'string' ? `${text} (in ${data.node_type})` : text;
  }
  return 'ComfyUI reported an error but gave no reason.';
}

async function getHistory(promptId: string, signal: AbortSignal): Promise<HistoryEntry | null> {
  const resp = await comfyRequest(`/history/${promptId}`, { signal });
  const data = (await resp.json()) as Record<string, HistoryEntry>;
  return data[promptId] ?? null;
}

/** setTimeout that rejects immediately on abort instead of only checking the signal after the
 * fact, so a cancel doesn't have to wait out the rest of the current poll interval. */
function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new GenerationCancelledError('Generation cancelled.'));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new GenerationCancelledError('Generation cancelled.'));
      },
      { once: true }
    );
  });
}

// A hard ceiling exists as a last-resort sanity net, but cancelCurrentGeneration() (a real
// Cancel button, not a guess) is the actual mechanism now - a slow but genuinely still-running
// generation (a cold model load, a long video render) shouldn't get killed by an arbitrary
// timeout the way a 10-minute default previously did, especially since that failure mode left
// the GPU still grinding on a job the app had already given up tracking.
async function waitForResult(
  promptId: string,
  signal: AbortSignal,
  timeoutMs = 3_600_000,
  pollIntervalMs = 1000
): Promise<HistoryEntry> {
  const start = Date.now();
  for (;;) {
    const entry = await getHistory(promptId, signal);
    if (entry) return entry;
    if (Date.now() - start > timeoutMs) {
      throw new Error(`ComfyUI prompt ${promptId} did not finish within ${timeoutMs}ms`);
    }
    await abortableSleep(pollIntervalMs, signal);
  }
}

async function fetchFileBytes(file: HistoryFile, signal: AbortSignal): Promise<Buffer> {
  const params = new URLSearchParams({
    filename: file.filename,
    subfolder: file.subfolder,
    type: file.type,
  });
  const resp = await comfyRequest(`/view?${params.toString()}`, { signal });
  return Buffer.from(await resp.arrayBuffer());
}

// Different ComfyUI save nodes have used different output key names across versions/node
// packs (SaveImage: "images", the native SaveVideo node and older VHS-style video nodes:
// "videos" or "gifs") - checked in priority order rather than assumed, since this hasn't
// been run against a real ComfyUI + Wan2.2 instance to confirm the exact key.
const OUTPUT_KEYS = ['images', 'videos', 'gifs'];

function extractOutputFile(entry: HistoryEntry, promptId: string): HistoryFile {
  const outputs = entry.outputs ?? {};
  for (const nodeOutput of Object.values(outputs)) {
    for (const key of OUTPUT_KEYS) {
      const files = nodeOutput[key];
      if (Array.isArray(files) && files.length > 0) return files[0];
    }
  }
  const failure = describeExecutionError(entry);
  if (failure) throw new Error(`ComfyUI could not run it: ${failure}`);
  throw new Error(`ComfyUI prompt ${promptId} finished with no image/video output`);
}

export interface GenerateOutput {
  bytes: Buffer;
  /** File extension including the leading dot, taken from ComfyUI's own output filename
   * (e.g. '.png', '.mp4') so the caller doesn't have to guess it per family. */
  extension: string;
  /** How the run's time split into loading / sampling / finishing (parts null if ComfyUI's live
   * events were unavailable). */
  timings: RunTimings;
}

interface ProgressSocket {
  /** Resolves once the socket is open (or has failed / timed out - progress is best-effort). */
  ready: Promise<void>;
  close: () => void;
}

/** Opens ComfyUI's websocket for `clientId`, which is where it reports what a job is doing. Only
 * jobs submitted with the same client id are reported on it, so it must be open before submitting.
 * Best-effort: the generation itself never depends on it. */
function openProgressSocket(clientId: string, onMessage: (message: ComfyMessage) => void): ProgressSocket | null {
  try {
    const wsBase = getEffectiveComfyUIHost().replace(/^http/i, 'ws');
    const ws = new WebSocket(`${wsBase}/ws?clientId=${clientId}`);
    ws.addEventListener('message', (event) => {
      // Binary frames are image previews; the JSON text frames carry progress.
      if (typeof event.data !== 'string') return;
      try {
        onMessage(JSON.parse(event.data) as ComfyMessage);
      } catch {
        // Ignore anything that is not a well-formed message.
      }
    });
    ws.addEventListener('error', () => undefined);
    const ready = new Promise<void>((resolve) => {
      const done = () => resolve();
      ws.addEventListener('open', done);
      ws.addEventListener('error', done);
      ws.addEventListener('close', done);
      setTimeout(done, 2000);
    });
    return {
      ready,
      close: () => {
        try {
          ws.close();
        } catch {
          // Already closed.
        }
      },
    };
  } catch {
    return null;
  }
}

function nodeClassesOf(workflow: Record<string, unknown>): Record<string, string> {
  const classes: Record<string, string> = {};
  for (const [id, node] of Object.entries(workflow)) {
    const cls = (node as { class_type?: unknown }).class_type;
    if (typeof cls === 'string') classes[id] = cls;
  }
  return classes;
}

/** Submit a generation and return the resulting file's bytes and extension. Only one
 * generation can be in flight at a time - see cancelCurrentGeneration(). `onProgress` receives
 * live stage/step updates from ComfyUI while it works. */
export async function generate(
  family: string,
  params: GenerationParams,
  onProgress?: (progress: GenerationProgress) => void
): Promise<GenerateOutput> {
  const startedAt = Date.now();
  const controller = new AbortController();
  currentAbortController = controller;
  let socket: ProgressSocket | null = null;
  try {
    const template = loadTemplate(family);
    const workflow = await patchTemplate(family, template, params, controller.signal);

    const tracker = new ProgressTracker(nodeClassesOf(workflow), startedAt);
    const clientId = randomUUID();
    socket = openProgressSocket(clientId, (message) => {
      const progress = tracker.handle(message);
      if (progress) onProgress?.(progress);
    });
    await socket?.ready;
    onProgress?.(tracker.snapshot());

    const promptId = await submit(workflow, controller.signal, clientId);
    currentPromptId = promptId;
    const replayed = tracker.setPromptId(promptId);
    if (replayed) onProgress?.(replayed);
    const entry = await waitForResult(promptId, controller.signal);
    const timings = tracker.finish();

    const file = extractOutputFile(entry, promptId);
    const bytes = await fetchFileBytes(file, controller.signal);
    const extension = path.extname(file.filename) || '.png';
    return { bytes, extension, timings };
  } finally {
    socket?.close();
    currentPromptId = null;
    currentAbortController = null;
  }
}
