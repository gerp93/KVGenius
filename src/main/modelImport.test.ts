import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { ModelImportError, checkModelFile, importModelFile } from './modelImport';
import { writeSafetensors } from './safetensors.test';

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-imp-'));
}

function setup() {
  const root = tempDir();
  const modelsDir = path.join(root, 'models');
  for (const sub of ['diffusion_models', 'vae', 'text_encoders', 'loras', 'upscale_models']) fs.mkdirSync(path.join(modelsDir, sub), { recursive: true });
  const downloads = path.join(root, 'downloads');
  fs.mkdirSync(downloads);
  return { root, modelsDir, downloads, cleanup: () => fs.rmSync(root, { recursive: true, force: true }) };
}

const layout = { 'a.weight': { shape: [8, 8] }, 'b.weight': { shape: [4] }, 'c.weight': { shape: [2, 2] } };

test('a file with the same layout as the known-good one checks out ok', async () => {
  const t = setup();
  try {
    writeSafetensors(path.join(t.modelsDir, 'vae', 'ae.safetensors'), layout);
    const candidate = path.join(t.downloads, 'ae-v2.safetensors');
    writeSafetensors(candidate, layout);
    const check = await checkModelFile(candidate, { folder: 'vae', modelsDir: t.modelsDir, referenceFile: 'ae.safetensors' });
    assert.equal(check.severity, 'ok');
    assert.equal(check.comparison?.verdict, 'match');
    assert.match(check.messages[0], /Same layout/);
  } finally {
    t.cleanup();
  }
});

test('a different layout warns, a cut-off download blocks', async () => {
  const t = setup();
  try {
    writeSafetensors(path.join(t.modelsDir, 'vae', 'ae.safetensors'), layout);
    const other = path.join(t.downloads, 'other.safetensors');
    writeSafetensors(other, { 'model.x': { shape: [1] }, 'model.y': { shape: [1] } });
    const different = await checkModelFile(other, { folder: 'vae', modelsDir: t.modelsDir, referenceFile: 'ae.safetensors' });
    assert.equal(different.severity, 'warn');
    assert.match(different.messages[0], /Does not look like ae\.safetensors/);

    const cut = path.join(t.downloads, 'cut.safetensors');
    writeSafetensors(cut, layout, 10);
    const incomplete = await checkModelFile(cut, { folder: 'vae', modelsDir: t.modelsDir, referenceFile: 'ae.safetensors' });
    assert.equal(incomplete.severity, 'block');
    assert.match(incomplete.messages[0], /incomplete/);
  } finally {
    t.cleanup();
  }
});

test('without a known-good file the check says so and does not guess', async () => {
  const t = setup();
  try {
    const candidate = path.join(t.downloads, 'x.safetensors');
    writeSafetensors(candidate, layout);
    for (const options of [
      { folder: 'vae' as const, modelsDir: null, referenceFile: 'ae.safetensors' },
      { folder: 'vae' as const, modelsDir: t.modelsDir, referenceFile: 'ae.safetensors' },
    ]) {
      const check = await checkModelFile(candidate, options);
      assert.equal(check.severity, 'ok');
      assert.equal(check.comparison, null);
      assert.match(check.messages[0], /Not compared/);
    }
  } finally {
    t.cleanup();
  }
});

test('other formats: not a model, GGUF, and the older code-carrying ones', async () => {
  const t = setup();
  try {
    const opts = { folder: 'diffusion_models' as const, modelsDir: null, referenceFile: null };
    const txt = path.join(t.downloads, 'notes.txt');
    fs.writeFileSync(txt, 'x');
    assert.equal((await checkModelFile(txt, opts)).severity, 'block');
    const gguf = path.join(t.downloads, 'm.gguf');
    fs.writeFileSync(gguf, 'x');
    assert.match((await checkModelFile(gguf, opts)).messages[0], /GGUF/);
    const ckpt = path.join(t.downloads, 'm.ckpt');
    fs.writeFileSync(ckpt, 'x');
    const warned = await checkModelFile(ckpt, opts);
    assert.equal(warned.severity, 'warn');
    assert.match(warned.messages[0], /run code/);
    const pth = path.join(t.downloads, 'up.pth');
    fs.writeFileSync(pth, 'x');
    const upscale = await checkModelFile(pth, { ...opts, folder: 'upscale_models' });
    assert.equal(upscale.severity, 'ok');
    assert.match(upscale.messages.join(' '), /Upscale page/);
    assert.equal((await checkModelFile(path.join(t.downloads, 'gone.safetensors'), opts)).severity, 'block');
  } finally {
    t.cleanup();
  }
});

test('importing copies the file under its own name into the right folder, with progress', async () => {
  const t = setup();
  try {
    const src = path.join(t.downloads, 'photoreal.safetensors');
    writeSafetensors(src, layout);
    const seen: number[] = [];
    const result = await importModelFile(src, { modelsDir: t.modelsDir, folder: 'diffusion_models', onProgress: (c) => seen.push(c) });
    assert.equal(result.destPath, path.join(t.modelsDir, 'diffusion_models', 'photoreal.safetensors'));
    assert.deepEqual(fs.readFileSync(result.destPath), fs.readFileSync(src));
    assert.ok(fs.existsSync(src), 'a copy leaves the original');
    assert.equal(seen[seen.length - 1], result.bytes);
    assert.deepEqual(fs.readdirSync(path.join(t.modelsDir, 'diffusion_models')), ['photoreal.safetensors']);
  } finally {
    t.cleanup();
  }
});

test('an existing file is not overwritten unless asked', async () => {
  const t = setup();
  try {
    const src = path.join(t.downloads, 'm.safetensors');
    writeSafetensors(src, layout);
    fs.writeFileSync(path.join(t.modelsDir, 'vae', 'm.safetensors'), 'old');
    await assert.rejects(() => importModelFile(src, { modelsDir: t.modelsDir, folder: 'vae' }), (e: unknown) => e instanceof ModelImportError && e.code === 'exists');
    assert.equal(fs.readFileSync(path.join(t.modelsDir, 'vae', 'm.safetensors'), 'utf8'), 'old');
    await importModelFile(src, { modelsDir: t.modelsDir, folder: 'vae', overwrite: true });
    assert.deepEqual(fs.readFileSync(path.join(t.modelsDir, 'vae', 'm.safetensors')), fs.readFileSync(src));
  } finally {
    t.cleanup();
  }
});

test('moving removes the original once the copy is in place', async () => {
  const t = setup();
  try {
    const src = path.join(t.downloads, 'm.safetensors');
    writeSafetensors(src, layout);
    const result = await importModelFile(src, { modelsDir: t.modelsDir, folder: 'vae', move: true });
    assert.equal(result.originalKept, false);
    assert.equal(fs.existsSync(src), false);
    assert.ok(fs.existsSync(result.destPath));
  } finally {
    t.cleanup();
  }
});

test('a cancelled import leaves nothing behind', async () => {
  const t = setup();
  try {
    const src = path.join(t.downloads, 'm.safetensors');
    writeSafetensors(src, layout);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(() => importModelFile(src, { modelsDir: t.modelsDir, folder: 'vae', signal: controller.signal }), (e: unknown) => e instanceof ModelImportError && e.code === 'cancelled');
    assert.deepEqual(fs.readdirSync(path.join(t.modelsDir, 'vae')), []);
  } finally {
    t.cleanup();
  }
});

test('a file already in place is left alone, and bad targets are refused', async () => {
  const t = setup();
  try {
    const inPlace = path.join(t.modelsDir, 'vae', 'ae.safetensors');
    writeSafetensors(inPlace, layout);
    const r = await importModelFile(inPlace, { modelsDir: t.modelsDir, folder: 'vae', move: true });
    assert.equal(r.destPath, inPlace);
    assert.ok(fs.existsSync(inPlace), 'a move onto itself must not delete it');

    const src = path.join(t.downloads, 'm.safetensors');
    writeSafetensors(src, layout);
    await assert.rejects(() => importModelFile(src, { modelsDir: t.modelsDir, folder: '../escape' as never }), /not a model folder/);
    await assert.rejects(() => importModelFile(src, { modelsDir: path.join(t.root, 'nope'), folder: 'vae' }), /not set or does not exist/);
    const txt = path.join(t.downloads, 'notes.txt');
    fs.writeFileSync(txt, 'x');
    await assert.rejects(() => importModelFile(txt, { modelsDir: t.modelsDir, folder: 'vae' }), /cannot be imported/);
  } finally {
    t.cleanup();
  }
});
