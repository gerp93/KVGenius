import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { guessModelsDir, looksLikeModelsDir, scanModelsDir } from './modelsFolder';

function tempRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-models-'));
}

function makeModels(dir: string): string {
  for (const sub of ['diffusion_models', 'vae', 'text_encoders', 'loras', 'upscale_models']) fs.mkdirSync(path.join(dir, sub), { recursive: true });
  return dir;
}

test('a folder with ComfyUI subfolders looks like a models folder; others do not', () => {
  const root = tempRoot();
  try {
    assert.equal(looksLikeModelsDir(path.join(root, 'nope')), false);
    assert.equal(looksLikeModelsDir(root), false);
    assert.equal(looksLikeModelsDir(makeModels(path.join(root, 'models'))), true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a portable install is found from its run script', () => {
  const root = tempRoot();
  try {
    const models = makeModels(path.join(root, 'ComfyUI_windows_portable', 'ComfyUI', 'models'));
    const script = path.join(root, 'ComfyUI_windows_portable', 'run_nvidia_gpu.bat');
    assert.equal(guessModelsDir(script, { home: path.join(root, 'nohome') }), models);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a script inside the ComfyUI folder finds the models beside it', () => {
  const root = tempRoot();
  try {
    const models = makeModels(path.join(root, 'ComfyUI', 'models'));
    assert.equal(guessModelsDir(path.join(root, 'ComfyUI', 'run.sh'), { home: path.join(root, 'nohome') }), models);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('Desktop falls back to the usual base folder, and a wrong guess is not offered', () => {
  const root = tempRoot();
  try {
    assert.equal(guessModelsDir('/Apps/ComfyUI.exe', { home: root }), null);
    const models = makeModels(path.join(root, 'Documents', 'ComfyUI', 'models'));
    assert.equal(guessModelsDir('/Apps/ComfyUI.exe', { home: root }), models);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a scan lists model files, with subfolders, and ignores other files', () => {
  const root = tempRoot();
  try {
    const models = makeModels(path.join(root, 'models'));
    fs.writeFileSync(path.join(models, 'vae', 'ae.safetensors'), 'x');
    fs.mkdirSync(path.join(models, 'vae', 'extra'));
    fs.writeFileSync(path.join(models, 'vae', 'extra', 'other.safetensors'), 'x');
    fs.writeFileSync(path.join(models, 'vae', 'readme.txt'), 'x');
    fs.writeFileSync(path.join(models, 'upscale_models', '4x.pth'), 'x');
    const installed = scanModelsDir(models);
    assert.deepEqual(installed.vae, ['ae.safetensors', 'extra/other.safetensors']);
    assert.deepEqual(installed.upscale_models, ['4x.pth']);
    assert.deepEqual(installed.loras, []);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
