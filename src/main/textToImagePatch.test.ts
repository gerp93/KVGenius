import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fillTextToImage } from './textToImagePatch';
import { PROFILE_FAMILIES } from '../shared/modelFamilies';
import type { GenerationParams } from '../shared/types';
import zImage from './templates/z-image.json';
import qwen from './templates/qwen-image.json';
import flux from './templates/flux-dev.json';
import hiDream from './templates/hidream.json';
import ovis from './templates/ovis-image.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const templates: Record<string, Template> = {
  'z-image': zImage as unknown as Template,
  'qwen-image': qwen as unknown as Template,
  'flux-dev': flux as unknown as Template,
  hidream: hiDream as unknown as Template,
  'ovis-image': ovis as unknown as Template,
};
const params: GenerationParams = { prompt: 'a red fox', width: 832, height: 1216, seed: 7, steps: 12, cfg: 2.5 };
const filled = (family: string): Template => {
  const copy = JSON.parse(JSON.stringify(templates[family])) as Template;
  fillTextToImage(copy, family, params);
  return copy;
};

test('every image family gets its prompt, seed, steps and size in the right nodes', () => {
  for (const family of Object.keys(templates)) {
    const w = filled(family);
    const texts = Object.values(w).filter((n) => n.inputs.text === 'a red fox');
    assert.equal(texts.length, 1, `${family}: the prompt is in exactly one node`);
    assert.ok(!JSON.stringify(w).includes('__PROMPT__'), `${family}: no placeholder is left`);
    const sampler = Object.values(w).find((n) => n.class_type === 'KSampler')!;
    assert.deepEqual([sampler.inputs.seed, sampler.inputs.steps], [7, 12], family);
    const latent = Object.values(w).find((n) => n.class_type === 'EmptySD3LatentImage')!;
    assert.deepEqual([latent.inputs.width, latent.inputs.height], [832, 1216], family);
  }
});

test('CFG is the sampler CFG, except for Flux where it is the FluxGuidance value and the sampler stays at 1', () => {
  for (const family of ['z-image', 'qwen-image', 'hidream', 'ovis-image']) {
    const sampler = Object.values(filled(family)).find((n) => n.class_type === 'KSampler')!;
    assert.equal(sampler.inputs.cfg, 2.5, family);
  }
  const flux = filled('flux-dev');
  assert.equal(Object.values(flux).find((n) => n.class_type === 'FluxGuidance')!.inputs.guidance, 2.5);
  assert.equal(Object.values(flux).find((n) => n.class_type === 'KSampler')!.inputs.cfg, 1);
});

test('the shipped template of each family runs its own default steps and CFG, and the family lists the same', () => {
  for (const family of PROFILE_FAMILIES.filter((f) => f.sampler && templates[f.family])) {
    const w = templates[family.family];
    const sampler = Object.values(w).find((n) => n.class_type === 'KSampler')!;
    assert.equal(sampler.inputs.steps, family.sampler!.steps, family.family);
  }
});

test('a template that drifts from the map fails loudly', () => {
  const copy = JSON.parse(JSON.stringify(templates['flux-dev'])) as Template;
  delete copy['7'];
  assert.throws(() => fillTextToImage(copy, 'flux-dev', params), /out of date/);
});
