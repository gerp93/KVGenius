import { randomUUID } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { GenerationParams } from '../shared/types';
import { getEffectiveComfyUIHost } from './dbLocation';
import zImageTurboTemplate from './templates/z-image-turbo.json';
import wan22I2vTemplate from './templates/wan22-i2v.json';

// Imported directly (not read from disk at runtime via fs) so tsc inlines the JSON into the
// compiled output - `tsc -p tsconfig.main.json` only compiles .ts files, it doesn't copy
// arbitrary assets into dist/, so a fs.readFileSync(path.join(__dirname, ...)) here would
// silently work in dev (where src/ and dist/ can end up looking similar) and fail in a real
// build with ENOENT once the template stops existing next to the compiled .js. One entry per
// model family template; add to this map as more templates are added.
const TEMPLATES: Record<string, Record<string, unknown>> = {
  'z-image-turbo': zImageTurboTemplate,
  'wan22-i2v': wan22I2vTemplate,
};

// ComfyUI Desktop (the Electron distribution this app targets) defaults to port 8000, not
// the classic standalone ComfyUI server's 8188 - different implementations, different defaults.
export const DEFAULT_COMFYUI_HOST = 'http://localhost:8000';

/**
 * Node IDs in src/main/templates/z-image-turbo.json that patchTemplate() fills in.
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
 * that the template ships already enabled - deliberately not exposed here, matching the
 * curated-per-family-template philosophy (see CLAUDE.md): only the fields a user actually
 * needs to touch are patched, everything else stays exactly as the template author set it.
 * Only samplerHighNoise's seed is patched - samplerLowNoise (129:85) has add_noise:'disable'
 * and return_with_leftover_noise from stage 1, so its own noise_seed field is inert.
 */
const WAN22_I2V_NODE_MAP = {
  loadImage: '97',
  positivePrompt: '129:93',
  imageToVideo: '129:98',
  samplerHighNoise: '129:86',
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
    throw new ComfyUIUnavailableError(`ComfyUI not reachable at ${host}: ${String(err)}`);
  }
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new ComfyUIUnavailableError(`ComfyUI returned ${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp;
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
  const template = TEMPLATES[family];
  if (!template) {
    throw new Error(`No ComfyUI workflow template registered for family '${family}'`);
  }
  return template;
}

/** Uploads a local image file to ComfyUI's input directory so a LoadImage node can
 * reference it by filename. Returns the filename ComfyUI stored it under. */
async function uploadSourceImage(filePath: string): Promise<string> {
  const bytes = fs.readFileSync(filePath);
  const form = new FormData();
  form.append('image', new Blob([bytes]), path.basename(filePath));
  const resp = await comfyRequest('/upload/image', { method: 'POST', body: form });
  const data = (await resp.json()) as { name?: string };
  if (!data.name) {
    throw new ComfyUIUnavailableError(`ComfyUI did not return a filename for the uploaded image: ${JSON.stringify(data)}`);
  }
  return data.name;
}

async function patchTemplate(
  family: string,
  template: Record<string, unknown>,
  params: GenerationParams
): Promise<Record<string, unknown>> {
  const workflow = JSON.parse(JSON.stringify(template));

  if (family === 'wan22-i2v') {
    if (!params.sourceImagePath) {
      throw new Error('Video mode requires a source image.');
    }
    const uploadedName = await uploadSourceImage(params.sourceImagePath);

    const loadImageNode = workflow[WAN22_I2V_NODE_MAP.loadImage] as { inputs: Record<string, unknown> };
    loadImageNode.inputs.image = uploadedName;

    const promptNode = workflow[WAN22_I2V_NODE_MAP.positivePrompt] as { inputs: Record<string, unknown> };
    promptNode.inputs.text = params.prompt;

    const imageToVideoNode = workflow[WAN22_I2V_NODE_MAP.imageToVideo] as { inputs: Record<string, unknown> };
    imageToVideoNode.inputs.width = params.width;
    imageToVideoNode.inputs.height = params.height;
    imageToVideoNode.inputs.length = params.length ?? 81;

    const samplerNode = workflow[WAN22_I2V_NODE_MAP.samplerHighNoise] as { inputs: Record<string, unknown> };
    samplerNode.inputs.noise_seed = params.seed;

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

  return workflow;
}

async function submit(workflow: Record<string, unknown>): Promise<string> {
  const resp = await comfyRequest('/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: workflow, client_id: randomUUID() }),
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
}

async function getHistory(promptId: string): Promise<HistoryEntry | null> {
  const resp = await comfyRequest(`/history/${promptId}`);
  const data = (await resp.json()) as Record<string, HistoryEntry>;
  return data[promptId] ?? null;
}

async function waitForResult(
  promptId: string,
  timeoutMs = 600_000,
  pollIntervalMs = 1000
): Promise<HistoryEntry> {
  const start = Date.now();
  for (;;) {
    const entry = await getHistory(promptId);
    if (entry) return entry;
    if (Date.now() - start > timeoutMs) {
      throw new Error(`ComfyUI prompt ${promptId} did not finish within ${timeoutMs}ms`);
    }
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }
}

async function fetchFileBytes(file: HistoryFile): Promise<Buffer> {
  const params = new URLSearchParams({
    filename: file.filename,
    subfolder: file.subfolder,
    type: file.type,
  });
  const resp = await comfyRequest(`/view?${params.toString()}`, { signal: AbortSignal.timeout(600_000) });
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
  throw new Error(`ComfyUI prompt ${promptId} finished with no image/video output`);
}

export interface GenerateOutput {
  bytes: Buffer;
  /** File extension including the leading dot, taken from ComfyUI's own output filename
   * (e.g. '.png', '.mp4') so the caller doesn't have to guess it per family. */
  extension: string;
}

/** Submit a generation and return the resulting file's bytes and extension. */
export async function generate(family: string, params: GenerationParams): Promise<GenerateOutput> {
  const template = loadTemplate(family);
  const workflow = await patchTemplate(family, template, params);
  const promptId = await submit(workflow);
  const entry = await waitForResult(promptId);

  const file = extractOutputFile(entry, promptId);
  const bytes = await fetchFileBytes(file);
  const extension = path.extname(file.filename) || '.png';
  return { bytes, extension };
}
