import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PROFILE_FAMILIES } from '../shared/modelFamilies';
import { ModelSettings, profileSettings } from '../shared/modelProfiles';
import { SAMPLER_NODES, SLOT_NODES, applyModelSettings } from './modelPatch';
import zImageTemplate from './templates/z-image.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const templates: Record<string, Template> = { 'z-image': zImageTemplate as unknown as Template };

test('every profile slot is wired to a template node whose current file is the slot default', () => {
  for (const family of PROFILE_FAMILIES) {
    const template = templates[family.family];
    assert.ok(template, `a template for ${family.family}`);
    for (const slot of family.slots) {
      const target = SLOT_NODES[family.family][slot.key];
      assert.ok(target, `${family.family}.${slot.key} has a node`);
      assert.equal(template[target.node].inputs[target.input], slot.defaultFile, `${family.family}.${slot.key} default`);
    }
  }
});

test('the sampler nodes are the ones that carry the shipped sampler values', () => {
  for (const family of PROFILE_FAMILIES) {
    if (!family.sampler) continue;
    const template = templates[family.family];
    const nodes = SAMPLER_NODES[family.family];
    const ksampler = template[nodes.sampler].inputs;
    assert.equal(ksampler.steps, family.sampler.steps);
    assert.equal(ksampler.cfg, family.sampler.cfg);
    assert.equal(ksampler.sampler_name, family.sampler.sampler);
    assert.equal(ksampler.scheduler, family.sampler.scheduler);
    assert.equal(template[nodes.shift].inputs.shift, family.sampler.shift);
  }
});

test('applying settings changes the files and sampler values, on a copy', () => {
  const copy = JSON.parse(JSON.stringify(zImageTemplate)) as Template;
  const settings: ModelSettings = {
    files: { diffusionModel: 'sub/photoreal.safetensors', textEncoder: 'enc.safetensors', vae: 'vae2.safetensors' },
    sampler: 'euler',
    scheduler: 'karras',
    shift: 5,
  };
  applyModelSettings(copy, 'z-image', settings);
  assert.equal(copy['57:28'].inputs.unet_name, 'sub/photoreal.safetensors');
  assert.equal(copy['57:30'].inputs.clip_name, 'enc.safetensors');
  assert.equal(copy['57:29'].inputs.vae_name, 'vae2.safetensors');
  assert.equal(copy['57:3'].inputs.sampler_name, 'euler');
  assert.equal(copy['57:3'].inputs.scheduler, 'karras');
  assert.equal(copy['57:11'].inputs.shift, 5);
  // the shipped template is untouched
  assert.equal((zImageTemplate as unknown as Template)['57:28'].inputs.unet_name, 'z_image_turbo_bf16.safetensors');
});

test('the built-in profile applied to the template changes nothing', () => {
  const family = PROFILE_FAMILIES[0];
  const copy = JSON.parse(JSON.stringify(zImageTemplate)) as Template;
  const builtIn = profileSettings({ files: Object.fromEntries(family.slots.map((s) => [s.key, s.defaultFile])), sampler: family.sampler! });
  applyModelSettings(copy, 'z-image', builtIn);
  assert.deepEqual(copy, zImageTemplate);
});

test('a missing file or an unsupported family fails loudly', () => {
  const copy = JSON.parse(JSON.stringify(zImageTemplate)) as Template;
  assert.throws(() => applyModelSettings(copy, 'z-image', { files: {}, sampler: 'a', scheduler: 'b', shift: 1 }), /image model/i);
  assert.throws(() => applyModelSettings(copy, 'wan22-i2v', { files: {}, sampler: 'a', scheduler: 'b', shift: 1 }), /not supported/);
});
