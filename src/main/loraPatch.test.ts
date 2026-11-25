import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LORA_LOADER_CLASS, LORA_NODE_MAP, applyLoras } from './loraPatch';
import { fillTextToImage } from './textToImagePatch';
import { LORA_FAMILIES, MAX_LORAS } from '../shared/loras';
import { FOLDER_LOADERS } from '../shared/modelManifest';
import flux from './templates/flux-dev.json';
import zImage from './templates/z-image.json';
import type { GenerationParams } from '../shared/types';

type Workflow = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const params: GenerationParams = { prompt: 'a twi\'lek dancer', width: 1024, height: 1024, seed: 3, steps: 20, cfg: 3, };
const fresh = (template: unknown = flux): Workflow => JSON.parse(JSON.stringify(template)) as Workflow;
const loraNodes = (w: Workflow) => Object.entries(w).filter(([, n]) => n.class_type === LORA_LOADER_CLASS);

test('the LoRA map agrees with the flux template, and every family that can load a LoRA has an entry', () => {
  for (const family of LORA_FAMILIES) assert.ok(LORA_NODE_MAP[family], `${family} has a LoRA node map`);
  const { model, consumers } = LORA_NODE_MAP['flux-dev'];
  const w = fresh();
  assert.equal(w[model].class_type, 'UNETLoader', 'the model node is the diffusion model loader');
  for (const { node, input } of consumers) assert.deepEqual(w[node].inputs[input], [model, 0], `${node}.${input} is fed by the model`);
  assert.equal(w['10'].class_type, 'KSampler');
  // The template itself holds no LoRA node: ComfyUI would refuse a LoRA name it does not have.
  assert.equal(loraNodes(w).length, 0);
  // The loader class is the one the Models page lists LoRA files from.
  assert.equal(FOLDER_LOADERS.loras.node, LORA_LOADER_CLASS);
});

test('with no LoRA the graph is untouched and has no LoRA node', () => {
  for (const loras of [undefined, []]) {
    const w = fresh();
    applyLoras(w, 'flux-dev', loras);
    assert.deepEqual(w, fresh());
    assert.equal(loraNodes(w).length, 0);
    assert.deepEqual(w['10'].inputs.model, ['2', 0], 'the sampler still takes the model directly');
  }
});

test('one LoRA sits between the model and the sampler, at its strength', () => {
  const w = fresh();
  applyLoras(w, 'flux-dev', [{ file: 'ffl-twilek.safetensors', strength: 0.8 }]);
  const nodes = loraNodes(w);
  assert.equal(nodes.length, 1);
  assert.deepEqual(nodes[0][1].inputs, { model: ['2', 0], lora_name: 'ffl-twilek.safetensors', strength_model: 0.8 });
  assert.deepEqual(w['10'].inputs.model, [nodes[0][0], 0]);
});

test('several LoRAs are chained in the order given, each at its own strength', () => {
  const w = fresh();
  applyLoras(w, 'flux-dev', [
    { file: 'a.safetensors', strength: 1 },
    { file: 'b.safetensors', strength: -0.5 },
    { file: 'sub/c.safetensors', strength: 0.25 },
  ]);
  assert.equal(loraNodes(w).length, 3);
  const [first, second, third] = ['lora-1', 'lora-2', 'lora-3'].map((id) => w[id]);
  assert.deepEqual(first.inputs, { model: ['2', 0], lora_name: 'a.safetensors', strength_model: 1 });
  assert.deepEqual(second.inputs, { model: ['lora-1', 0], lora_name: 'b.safetensors', strength_model: -0.5 });
  assert.deepEqual(third.inputs, { model: ['lora-2', 0], lora_name: 'sub/c.safetensors', strength_model: 0.25 });
  assert.deepEqual(w['10'].inputs.model, ['lora-3', 0], 'the sampler takes the last one');
});

test('no more than the maximum are loaded, and strengths are kept within range', () => {
  const w = fresh();
  const many = Array.from({ length: MAX_LORAS + 2 }, (_, i) => ({ file: `l${i}.safetensors`, strength: i === 0 ? 9 : 1 }));
  applyLoras(w, 'flux-dev', many);
  assert.equal(loraNodes(w).length, MAX_LORAS);
  assert.equal(w['lora-1'].inputs.strength_model, 2, 'out-of-range strength is clamped');
  assert.deepEqual(w['10'].inputs.model, [`lora-${MAX_LORAS}`, 0]);
});

test('a family that cannot load a LoRA ignores them', () => {
  const w = fresh(zImage);
  applyLoras(w, 'z-image', [{ file: 'x.safetensors', strength: 1 }]);
  assert.deepEqual(w, fresh(zImage));
});

test('it composes with the text-to-image fields, and a template that drifts fails loudly', () => {
  const w = fresh();
  fillTextToImage(w, 'flux-dev', params);
  applyLoras(w, 'flux-dev', [{ file: 'a.safetensors', strength: 1 }]);
  assert.equal(w['6'].inputs.text, params.prompt);
  assert.equal(w['7'].inputs.guidance, 3);
  assert.deepEqual(w['10'].inputs.model, ['lora-1', 0]);

  const missingModel = fresh();
  delete missingModel['2'];
  assert.throws(() => applyLoras(missingModel, 'flux-dev', [{ file: 'a.safetensors', strength: 1 }]), /out of date/);
  const rewired = fresh();
  rewired['10'].inputs.model = ['3', 0];
  assert.throws(() => applyLoras(rewired, 'flux-dev', [{ file: 'a.safetensors', strength: 1 }]), /out of date/);
  // With no LoRA there is nothing to patch, so a drifted template is not this patch's problem.
  assert.doesNotThrow(() => applyLoras(rewired, 'flux-dev', []));
});
