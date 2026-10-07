import * as zlib from 'zlib';
import type { ImageSettings } from '../shared/modelCheck';

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** The text chunks (tEXt, zTXt, iTXt) of a PNG, by keyword. Anything malformed ends the scan with what was found. */
export function readPngTextChunks(buffer: Buffer): Record<string, string> {
  const found: Record<string, string> = {};
  if (buffer.length < 8 || !buffer.subarray(0, 8).equals(PNG_SIGNATURE)) return found;
  let pos = 8;
  while (pos + 12 <= buffer.length) {
    const length = buffer.readUInt32BE(pos);
    const type = buffer.toString('latin1', pos + 4, pos + 8);
    const start = pos + 8;
    const end = start + length;
    if (end + 4 > buffer.length) break;
    const data = buffer.subarray(start, end);
    try {
      if (type === 'tEXt') {
        const nul = data.indexOf(0);
        if (nul > 0) found[data.toString('latin1', 0, nul)] = data.toString('utf8', nul + 1);
      } else if (type === 'zTXt') {
        const nul = data.indexOf(0);
        if (nul > 0) found[data.toString('latin1', 0, nul)] = zlib.inflateSync(data.subarray(nul + 2)).toString('utf8');
      } else if (type === 'iTXt') {
        const nul = data.indexOf(0);
        if (nul > 0) {
          const compressed = data[nul + 1] === 1;
          // keyword NUL, compression flag, method, language NUL, translated keyword NUL, text
          const langEnd = data.indexOf(0, nul + 3);
          const transEnd = langEnd < 0 ? -1 : data.indexOf(0, langEnd + 1);
          if (transEnd >= 0) {
            const body = data.subarray(transEnd + 1);
            found[data.toString('latin1', 0, nul)] = (compressed ? zlib.inflateSync(body) : body).toString('utf8');
          }
        }
      } else if (type === 'IEND') {
        break;
      }
    } catch {
      // A chunk that will not decode is skipped; the rest may still be fine.
    }
    pos = end + 4;
  }
  return found;
}

type ApiNode = { class_type?: unknown; inputs?: Record<string, unknown> };

function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Settings from the workflow ComfyUI saves in its pictures (the "prompt" chunk, in API format). */
function fromComfyPrompt(text: string): ImageSettings | null {
  let graph: Record<string, ApiNode>;
  try {
    graph = JSON.parse(text) as Record<string, ApiNode>;
  } catch {
    return null;
  }
  if (!graph || typeof graph !== 'object') return null;
  const nodes = Object.values(graph).filter((n): n is ApiNode => !!n && typeof n === 'object' && !!n.inputs);
  const sampler = nodes.find((n) => (n.class_type === 'KSampler' || n.class_type === 'KSamplerAdvanced') && num(n.inputs?.steps) !== undefined);
  if (!sampler?.inputs) return null;
  const result: ImageSettings = { source: 'comfyui' };
  result.steps = num(sampler.inputs.steps);
  result.cfg = num(sampler.inputs.cfg);
  result.sampler = str(sampler.inputs.sampler_name);
  result.scheduler = str(sampler.inputs.scheduler);
  const shift = nodes.find((n) => (n.class_type === 'ModelSamplingAuraFlow' || n.class_type === 'ModelSamplingSD3') && num(n.inputs?.shift) !== undefined);
  if (shift?.inputs) result.shift = num(shift.inputs.shift);
  const hints: NonNullable<ImageSettings['fileHints']> = {};
  hints.diffusionModel = str(nodes.find((n) => n.class_type === 'UNETLoader')?.inputs?.unet_name);
  hints.textEncoder = str(nodes.find((n) => n.class_type === 'CLIPLoader')?.inputs?.clip_name);
  hints.vae = str(nodes.find((n) => n.class_type === 'VAELoader')?.inputs?.vae_name);
  if (hints.diffusionModel || hints.textEncoder || hints.vae) result.fileHints = hints;
  return result;
}

/** Steps and CFG from an Automatic1111-style "parameters" text. Its sampler names differ from ComfyUI's, so they are not read. */
function fromParameters(text: string): ImageSettings | null {
  const steps = /Steps:\s*(\d+)/.exec(text);
  const cfg = /CFG scale:\s*([\d.]+)/.exec(text);
  if (!steps && !cfg) return null;
  return { source: 'a1111', ...(steps ? { steps: Number(steps[1]) } : {}), ...(cfg ? { cfg: Number(cfg[1]) } : {}) };
}

/** What a PNG says about how it was made, or null when it carries nothing usable (most shared pictures have it stripped). */
export function settingsFromPng(buffer: Buffer): ImageSettings | null {
  const chunks = readPngTextChunks(buffer);
  if (chunks.prompt) {
    const fromComfy = fromComfyPrompt(chunks.prompt);
    if (fromComfy) return fromComfy;
  }
  if (chunks.parameters) return fromParameters(chunks.parameters);
  return null;
}
