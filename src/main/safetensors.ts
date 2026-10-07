import * as fs from 'fs';
import { TensorComparison } from '../shared/modelCheck';

export interface TensorInfo {
  dtype: string;
  shape: number[];
  /** [start, end) of its data, relative to the end of the header. */
  offsets: [number, number];
}

export interface SafetensorsHeader {
  tensors: Record<string, TensorInfo>;
  /** Bytes the file must have to hold the header and every tensor. */
  expectedSize: number;
}

/** safetensors' own sanity limit on the JSON header. */
const MAX_HEADER_BYTES = 100 * 1024 * 1024;

/**
 * Reads only the header of a .safetensors file: 8 bytes (little-endian length of the JSON), then the JSON,
 * which lists every tensor's name, type, shape and byte range. The tensors themselves are never read, so this is
 * instant even for a 14 GB file. Returns a reason when the file is not a valid safetensors file.
 */
export async function readSafetensorsHeader(filePath: string): Promise<{ ok: true; header: SafetensorsHeader; fileSize: number } | { ok: false; reason: string }> {
  let handle: fs.promises.FileHandle | null = null;
  try {
    handle = await fs.promises.open(filePath, 'r');
    const fileSize = (await handle.stat()).size;
    if (fileSize < 8) return { ok: false, reason: 'The file is too small to be a safetensors file.' };
    const lengthBytes = Buffer.alloc(8);
    await handle.read(lengthBytes, 0, 8, 0);
    const length = Number(lengthBytes.readBigUInt64LE(0));
    if (!Number.isSafeInteger(length) || length <= 0 || length > MAX_HEADER_BYTES || 8 + length > fileSize) {
      return { ok: false, reason: 'The file does not start like a safetensors file (its header length is not valid).' };
    }
    const json = Buffer.alloc(length);
    await handle.read(json, 0, length, 8);
    return parseSafetensorsHeader(json.toString('utf8'), length, fileSize);
  } catch (err) {
    return { ok: false, reason: `The file could not be read: ${err instanceof Error ? err.message : String(err)}` };
  } finally {
    await handle?.close();
  }
}

/** The pure half of readSafetensorsHeader, so it can be tested on built headers. */
export function parseSafetensorsHeader(json: string, headerLength: number, fileSize: number): { ok: true; header: SafetensorsHeader; fileSize: number } | { ok: false; reason: string } {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(json) as Record<string, unknown>;
  } catch {
    return { ok: false, reason: 'The file does not start like a safetensors file (its header is not readable).' };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return { ok: false, reason: 'The file does not start like a safetensors file.' };
  const tensors: Record<string, TensorInfo> = {};
  let dataEnd = 0;
  for (const [name, value] of Object.entries(parsed)) {
    if (name === '__metadata__') continue;
    // The file calls a tensor's byte range `data_offsets`: [start, end) within the data that follows the header.
    const t = value as { dtype?: unknown; shape?: unknown; data_offsets?: unknown } | null;
    if (!t || typeof t.dtype !== 'string' || !Array.isArray(t.shape) || !Array.isArray(t.data_offsets) || t.data_offsets.length !== 2) {
      return { ok: false, reason: `The tensor "${name}" in the file header is malformed.` };
    }
    const offsets: [number, number] = [Number(t.data_offsets[0]), Number(t.data_offsets[1])];
    if (!Number.isFinite(offsets[0]) || !Number.isFinite(offsets[1]) || offsets[0] < 0 || offsets[1] < offsets[0]) {
      return { ok: false, reason: `The tensor "${name}" in the file header has an invalid byte range.` };
    }
    tensors[name] = { dtype: t.dtype, shape: (t.shape as unknown[]).map(Number), offsets };
    dataEnd = Math.max(dataEnd, offsets[1]);
  }
  return { ok: true, header: { tensors, expectedSize: 8 + headerLength + dataEnd }, fileSize };
}

/** Per-tensor bookkeeping some quantised (fp8 "scaled") files add; it says nothing about the architecture. */
const BOOKKEEPING = /(^|\.)(scale_weight|scale_input|weight_scale|input_scale|comfy_quant|scaled_fp8)$/;

function coreNames(header: SafetensorsHeader): string[] {
  return Object.keys(header.tensors).filter((name) => !BOOKKEEPING.test(name));
}

function sameShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((n, i) => n === b[i]);
}

/**
 * Compares a candidate's tensor names and shapes (never their data types, so an fp8 and a bf16 copy of one model
 * still match) with a known-good file for the same slot. A fine-tune or merge keeps the layout; a different
 * architecture, or the same one packaged differently (say, an all-in-one checkpoint), does not. The thresholds are
 * a judgement, not a proof - the test render is the real check.
 */
export function compareHeaders(reference: SafetensorsHeader, candidate: SafetensorsHeader): TensorComparison {
  const ref = coreNames(reference);
  const cand = coreNames(candidate);
  const candSet = new Set(cand);
  const shared = ref.filter((name) => candSet.has(name));
  const shapeMismatches = shared.filter((name) => !sameShape(reference.tensors[name].shape, candidate.tensors[name].shape)).length;
  const coverage = ref.length === 0 ? 0 : shared.length / ref.length;
  const extra = cand.length === 0 ? 0 : (cand.length - shared.length) / cand.length;
  let verdict: TensorComparison['verdict'];
  if (coverage >= 0.98 && extra <= 0.02 && shapeMismatches === 0) verdict = 'match';
  else if (coverage >= 0.5) verdict = 'related';
  else verdict = 'different';
  return { verdict, referenceTensors: ref.length, candidateTensors: cand.length, shared: shared.length, shapeMismatches };
}
