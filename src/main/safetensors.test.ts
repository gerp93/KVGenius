import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { SafetensorsHeader, compareHeaders, parseSafetensorsHeader, readSafetensorsHeader } from './safetensors';

/** Builds a real .safetensors file (a tiny one: each tensor is `bytes` zero bytes). */
export function writeSafetensors(file: string, tensors: Record<string, { shape: number[]; dtype?: string }>, truncateBy = 0): void {
  const header: Record<string, unknown> = { __metadata__: { format: 'pt' } };
  let offset = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const bytes = t.shape.reduce((a, b) => a * b, 1) * 2;
    header[name] = { dtype: t.dtype ?? 'BF16', shape: t.shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  const json = Buffer.from(JSON.stringify(header), 'utf8');
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(json.length));
  const data = Buffer.alloc(Math.max(0, offset - truncateBy));
  fs.writeFileSync(file, Buffer.concat([length, json, data]));
}

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-st-'));
}

function header(tensors: Record<string, { shape: number[]; dtype?: string }>): SafetensorsHeader {
  let offset = 0;
  const out: SafetensorsHeader['tensors'] = {};
  for (const [name, t] of Object.entries(tensors)) {
    out[name] = { dtype: t.dtype ?? 'BF16', shape: t.shape, offsets: [offset, offset + 2] };
    offset += 2;
  }
  return { tensors: out, expectedSize: 0 };
}

test('a header is read without the tensors, and the expected size follows from it', async () => {
  const dir = tempDir();
  try {
    const file = path.join(dir, 'm.safetensors');
    writeSafetensors(file, { a: { shape: [2, 3] }, b: { shape: [4] } });
    const r = await readSafetensorsHeader(file);
    assert.ok(r.ok);
    if (!r.ok) return;
    assert.deepEqual(Object.keys(r.header.tensors), ['a', 'b']);
    assert.deepEqual(r.header.tensors.a.shape, [2, 3]);
    assert.equal(r.header.expectedSize, r.fileSize);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a truncated file is recognised by its size being below what the header promises', async () => {
  const dir = tempDir();
  try {
    const file = path.join(dir, 'm.safetensors');
    writeSafetensors(file, { a: { shape: [8, 8] } }, 20);
    const r = await readSafetensorsHeader(file);
    assert.ok(r.ok);
    if (r.ok) assert.ok(r.fileSize < r.header.expectedSize);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('files that are not safetensors are refused with a reason', async () => {
  const dir = tempDir();
  try {
    const tiny = path.join(dir, 'tiny.safetensors');
    fs.writeFileSync(tiny, 'abc');
    assert.match(String((await readSafetensorsHeader(tiny) as { reason?: string }).reason), /too small/);
    const text = path.join(dir, 'text.safetensors');
    fs.writeFileSync(text, 'this is not a model file at all, just text');
    assert.equal((await readSafetensorsHeader(text)).ok, false);
    assert.equal((await readSafetensorsHeader(path.join(dir, 'missing.safetensors'))).ok, false);
    assert.equal(parseSafetensorsHeader('{"a": 5}', 8, 100).ok, false);
    // The byte range must be the file's own `data_offsets`, and a sane one.
    assert.equal(parseSafetensorsHeader('{"a": {"dtype": "F16", "shape": [1], "offsets": [0, 2]}}', 40, 100).ok, false);
    assert.equal(parseSafetensorsHeader('{"a": {"dtype": "F16", "shape": [1], "data_offsets": [4, 2]}}', 40, 100).ok, false);
    assert.equal(parseSafetensorsHeader('[]', 2, 100).ok, false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a fine-tune with the same layout matches, even in another precision or with scale tensors', () => {
  const ref = header({ 'l1.weight': { shape: [4, 4] }, 'l2.weight': { shape: [8, 4] }, 'l3.weight': { shape: [2] } });
  const finetune = header({ 'l1.weight': { shape: [4, 4], dtype: 'F8_E4M3' }, 'l2.weight': { shape: [8, 4] }, 'l3.weight': { shape: [2] }, 'l1.scale_weight': { shape: [1] } });
  const r = compareHeaders(ref, finetune);
  assert.equal(r.verdict, 'match');
  assert.equal(r.shapeMismatches, 0);
  assert.equal(r.candidateTensors, 3);
});

test('a changed shape, missing tensors or a different layout are not a match', () => {
  const ref = header({ a: { shape: [4, 4] }, b: { shape: [4] }, c: { shape: [2] }, d: { shape: [2] } });
  assert.equal(compareHeaders(ref, header({ a: { shape: [4, 8] }, b: { shape: [4] }, c: { shape: [2] }, d: { shape: [2] } })).verdict, 'related');
  assert.equal(compareHeaders(ref, header({ a: { shape: [4, 4] }, b: { shape: [4] } })).verdict, 'related');
  const other = compareHeaders(ref, header({ 'model.x': { shape: [1] }, 'model.y': { shape: [1] } }));
  assert.equal(other.verdict, 'different');
  assert.equal(other.shared, 0);
});

test('extra tensors beyond a small allowance stop it being a match', () => {
  const ref = header({ a: { shape: [1] }, b: { shape: [1] } });
  const bigger = header({ a: { shape: [1] }, b: { shape: [1] }, c: { shape: [1] }, d: { shape: [1] }, e: { shape: [1] } });
  assert.notEqual(compareHeaders(ref, bigger).verdict, 'match');
});
