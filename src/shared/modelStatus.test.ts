import { test } from 'node:test';
import assert from 'node:assert/strict';
import { emptyInstalled, fileState, parseChoiceList, readinessLabel, summarize, summarizeSlots } from './modelStatus';
import { ManifestFile } from './modelManifest';

const vae: ManifestFile = { file: 'ae.safetensors', folder: 'vae', role: 'VAE' };
const unet: ManifestFile = { file: 'big.safetensors', folder: 'diffusion_models', role: 'Model' };

test('both shapes of ComfyUI choice list are read', () => {
  assert.deepEqual(parseChoiceList([['a.safetensors', 'b.safetensors']]), ['a.safetensors', 'b.safetensors']);
  assert.deepEqual(parseChoiceList(['COMBO', { options: ['c.pth'] }]), ['c.pth']);
});

test('anything else is no choices', () => {
  assert.deepEqual(parseChoiceList(undefined), []);
  assert.deepEqual(parseChoiceList('nope'), []);
  assert.deepEqual(parseChoiceList(['COMBO', {}]), []);
});

test('a file is present only under exactly the requested name', () => {
  const installed = emptyInstalled();
  installed.vae = ['ae.safetensors'];
  assert.equal(fileState(vae, { source: 'comfyui', installed }).state, 'present');
  assert.equal(fileState(unet, { source: 'comfyui', installed }).state, 'missing');
});

test('a file in a subfolder is called out, not counted as present', () => {
  const installed = emptyInstalled();
  installed.vae = ['my-stuff/ae.safetensors'];
  const state = fileState(vae, { source: 'comfyui', installed });
  assert.deepEqual(state, { state: 'in-subfolder', foundAs: 'my-stuff/ae.safetensors' });
  const windows = emptyInstalled();
  windows.vae = ['my-stuff\\ae.safetensors'];
  assert.equal(fileState(vae, { source: 'folder', installed: windows }).state, 'in-subfolder');
});

test('with no source of information nothing is claimed', () => {
  assert.equal(fileState(vae, { source: 'none', installed: emptyInstalled() }).state, 'unknown');
  assert.deepEqual(summarize([vae, unet], { source: 'none', installed: emptyInstalled() }), {
    total: 2,
    present: 0,
    missing: 0,
    inSubfolder: 0,
  });
});

test('a summary counts each state', () => {
  const installed = emptyInstalled();
  installed.vae = ['ae.safetensors'];
  installed.diffusion_models = ['x/big.safetensors'];
  assert.deepEqual(summarize([vae, unet], { source: 'comfyui', installed }), { total: 2, present: 1, missing: 0, inSubfolder: 1 });
});

test('a model\'s slots are summarised like a feature\'s files', () => {
  const report = { source: 'comfyui' as const, installed: { ...emptyInstalled(), diffusion_models: ['a.safetensors'], vae: ['v.safetensors'] } };
  const slots = [
    { key: 'model', folder: 'diffusion_models' as const },
    { key: 'vae', folder: 'vae' as const },
    { key: 'enc', folder: 'text_encoders' as const },
  ];
  const summary = summarizeSlots(slots, { model: 'a.safetensors', vae: 'v.safetensors', enc: 'e.safetensors' }, report);
  assert.equal(summary.present, 2);
  assert.equal(summary.missing, 1);
  assert.deepEqual(readinessLabel(summary), { ok: false, text: '1 file missing' });
  assert.deepEqual(readinessLabel(summarizeSlots(slots.slice(0, 2), { model: 'a.safetensors', vae: 'v.safetensors' }, report)), { ok: true, text: 'ready' });
  // a slot with no file chosen is missing, and nothing known gives no label
  assert.equal(summarizeSlots(slots.slice(0, 1), {}, report).missing, 1);
  assert.equal(readinessLabel(null), null);
});
