import { test } from 'node:test';
import assert from 'node:assert/strict';
import chroma from './templates/chroma.json';
import { fillTextToImage } from './textToImagePatch';
import { applyModelSettings } from './modelPatch';
import { applyLoras } from './loraPatch';
import { CHROMA_FAMILY } from '../shared/modelFamilies';
import { TEXT_TO_IMAGE_FAMILIES } from '../shared/families';
import { FAMILY_KIND } from '../shared/types';
import { generationOrigin } from '../shared/origin';
import { profileSettings } from '../shared/modelProfiles';
import type { GenerationParams } from '../shared/types';

type Workflow = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const fresh = () => JSON.parse(JSON.stringify(chroma)) as Workflow;
const params: GenerationParams = { prompt: 'a red fox', width: 832, height: 1216, seed: 7, steps: 30, cfg: 4 };

test('chroma is a registered text-to-image family', () => {
  assert.ok(TEXT_TO_IMAGE_FAMILIES.includes('chroma'));
  assert.equal(FAMILY_KIND.chroma, 'image');
  assert.equal(generationOrigin('chroma')?.kind, 'text-to-image');
});

test('the template follows ComfyUI\'s Chroma workflow: a chroma-type T5 loader through the tokenizer options, a negative prompt, a real CFG on a beta schedule', () => {
  const w = fresh();
  assert.equal(w['3'].class_type, 'CLIPLoader');
  assert.equal(w['3'].inputs.type, 'chroma');
  assert.equal(w['12'].class_type, 'T5TokenizerOptions');
  assert.deepEqual(w['12'].inputs.clip, ['3', 0]);
  // Both prompts read the text encoder through the tokenizer options.
  assert.deepEqual(w['6'].inputs.clip, ['12', 0]);
  assert.deepEqual(w['7'].inputs.clip, ['12', 0]);
  assert.ok(String(w['7'].inputs.text).trim().length > 0, 'the negative prompt ships with the template');
  // The model goes through the flow shift, and the sampler takes everything.
  assert.deepEqual(w['5'].inputs.model, ['2', 0]);
  const sampler: Record<string, unknown> = w['10'].inputs;
  for (const [input, link] of Object.entries({ model: ['5', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['9', 0] })) {
    assert.deepEqual(sampler[input], link, input);
  }
  assert.equal(w['10'].inputs.scheduler, 'beta');
  assert.equal(w['10'].inputs.cfg, CHROMA_FAMILY.sampler.cfg);
  assert.equal(w['10'].inputs.steps, CHROMA_FAMILY.sampler.steps);
  assert.equal(w['5'].inputs.shift, CHROMA_FAMILY.sampler.shift);
});

test('a run fills the prompt, seed, steps, CFG and size, and leaves the negative prompt alone', () => {
  const w = fresh();
  const negative = w['7'].inputs.text;
  fillTextToImage(w, 'chroma', params);
  assert.equal(w['6'].inputs.text, 'a red fox');
  assert.deepEqual([w['10'].inputs.seed, w['10'].inputs.steps, w['10'].inputs.cfg], [7, 30, 4], 'CFG is the sampler\'s own, as for the other full models');
  assert.deepEqual([w['9'].inputs.width, w['9'].inputs.height], [832, 1216]);
  assert.equal(w['7'].inputs.text, negative);
  assert.ok(!JSON.stringify(w).includes('__PROMPT__'));
});

test('a saved model swaps the three files and the sampler values', () => {
  const w = fresh();
  const settings = profileSettings({
    family: 'chroma',
    files: { diffusionModel: 'chroma-custom.safetensors', textEncoder: 't5-custom.safetensors', vae: 'ae2.safetensors' },
    sampler: { steps: 20, cfg: 5, sampler: 'euler', scheduler: 'simple', shift: 2 },
  });
  applyModelSettings(w, 'chroma', settings);
  assert.equal(w['2'].inputs.unet_name, 'chroma-custom.safetensors');
  assert.equal(w['3'].inputs.clip_name, 't5-custom.safetensors');
  assert.equal(w['4'].inputs.vae_name, 'ae2.safetensors');
  assert.equal(w['10'].inputs.scheduler, 'simple');
  assert.equal(w['5'].inputs.shift, 2);
});

test('Flux LoRAs are not applied to Chroma', () => {
  const w = fresh();
  applyLoras(w, 'chroma', [{ file: 'a.safetensors', strength: 1 }]);
  assert.deepEqual(w, fresh());
});
