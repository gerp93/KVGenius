import { randomUUID } from 'crypto';
import { GenerationParams } from '../shared/types';
import { getEffectiveComfyUIHost } from './dbLocation';
import zImageTurboTemplate from './templates/z-image-turbo.json';

// Imported directly (not read from disk at runtime via fs) so tsc inlines the JSON into the
// compiled output - `tsc -p tsconfig.main.json` only compiles .ts files, it doesn't copy
// arbitrary assets into dist/, so a fs.readFileSync(path.join(__dirname, ...)) here would
// silently work in dev (where src/ and dist/ can end up looking similar) and fail in a real
// build with ENOENT once the template stops existing next to the compiled .js. One entry per
// model family template; add to this map as more templates are added.
const TEMPLATES: Record<string, Record<string, unknown>> = {
  'z-image-turbo': zImageTurboTemplate,
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
const NODE_MAP = {
  prompt: '57:27',
  sampler: '57:3',
  latent: '57:13',
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

function patchTemplate(template: Record<string, unknown>, params: GenerationParams): Record<string, unknown> {
  const workflow = JSON.parse(JSON.stringify(template));

  const promptNode = workflow[NODE_MAP.prompt] as { inputs: Record<string, unknown> };
  promptNode.inputs.text = params.prompt;

  const samplerNode = workflow[NODE_MAP.sampler] as { inputs: Record<string, unknown> };
  samplerNode.inputs.seed = params.seed;
  samplerNode.inputs.steps = params.steps;
  samplerNode.inputs.cfg = params.cfg;

  const latentNode = workflow[NODE_MAP.latent] as { inputs: Record<string, unknown> };
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

interface HistoryImage {
  filename: string;
  subfolder: string;
  type: string;
}

interface HistoryEntry {
  outputs?: Record<string, { images?: HistoryImage[] }>;
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

async function fetchImageBytes(img: HistoryImage): Promise<Buffer> {
  const params = new URLSearchParams({
    filename: img.filename,
    subfolder: img.subfolder,
    type: img.type,
  });
  const resp = await comfyRequest(`/view?${params.toString()}`, { signal: AbortSignal.timeout(60_000) });
  return Buffer.from(await resp.arrayBuffer());
}

/** Submit a generation and return the resulting PNG bytes. */
export async function generate(family: string, params: GenerationParams): Promise<Buffer> {
  const template = loadTemplate(family);
  const workflow = patchTemplate(template, params);
  const promptId = await submit(workflow);
  const entry = await waitForResult(promptId);

  const outputs = entry.outputs ?? {};
  for (const nodeOutput of Object.values(outputs)) {
    const images = nodeOutput.images;
    if (images && images.length > 0) {
      return fetchImageBytes(images[0]);
    }
  }
  throw new Error(`ComfyUI prompt ${promptId} finished with no image output`);
}
