import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PROFILE_FAMILIES, profileFamilyKey } from '../shared/modelFamilies';
import { ModelSettings, profileSettings } from '../shared/modelProfiles';
import { SAMPLER_NODES, SLOT_NODES, applyModelSettings } from './modelPatch';
import zImageTemplate from './templates/z-image.json';
import i2iTemplate from './templates/z-image-i2i.json';
import inpaintTemplate from './templates/z-image-inpaint.json';
import outpaintTemplate from './templates/z-image-outpaint.json';
import wanTemplate from './templates/wan22-i2v.json';
import wanTextTemplate from './templates/wan22-t2v.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const templates: Record<string, Template> = { 'z-image': zImageTemplate as unknown as Template, 'z-image-i2i': i2iTemplate as unknown as Template, 'z-image-inpaint': inpaintTemplate as unknown as Template, 'z-image-outpaint': outpaintTemplate as unknown as Template, 'wan22-i2v': wanTemplate as unknown as Template, 'wan22-t2v': wanTextTemplate as unknown as Template };

test('every profile slot is wired to a template node whose current file is the slot default - in every template that uses the family', () => {
  for (const [templateFamily, template] of Object.entries(templates)) {
    // image to image runs Z-Image's files, so it is held to Z-Image's slots too
    const family = PROFILE_FAMILIES.find((f) => f.family === profileFamilyKey(templateFamily));
    assert.ok(family, `a profile family for ${templateFamily}`);
    for (const slot of family.slots) {
      const target: { node: string; input: string } | undefined = SLOT_NODES[family.family][slot.key];
      assert.ok(target, `${family.family}.${slot.key} has a node`);
      assert.equal(template[target.node].inputs[target.input], slot.defaultFile, `${templateFamily}.${slot.key} default`);
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
  const builtIn = profileSettings({ family: 'z-image', files: Object.fromEntries(family.slots.map((s) => [s.key, s.defaultFile])), sampler: family.sampler! });
  applyModelSettings(copy, 'z-image', builtIn);
  assert.deepEqual(copy, zImageTemplate);
});

test('a missing file or an unsupported family fails loudly', () => {
  const copy = JSON.parse(JSON.stringify(zImageTemplate)) as Template;
  assert.throws(() => applyModelSettings(copy, 'z-image', { files: {}, sampler: 'a', scheduler: 'b', shift: 1 }), /image model/i);
  assert.throws(() => applyModelSettings(copy, 'upscale-image', { files: {}, sampler: 'a', scheduler: 'b', shift: 1 }), /not supported/);
});

test('a video profile swaps all six Wan files and leaves the sampler alone', () => {
  const copy = JSON.parse(JSON.stringify(wanTemplate)) as Template;
  const before = JSON.stringify(copy);
  applyModelSettings(copy, 'wan22-i2v', {
    files: {
      highNoiseModel: 'hi.safetensors',
      lowNoiseModel: 'lo.safetensors',
      textEncoder: 'umt5.safetensors',
      vae: 'wanvae.safetensors',
      highNoiseLora: 'lora-hi.safetensors',
      lowNoiseLora: 'lora-lo.safetensors',
    },
  });
  assert.equal(copy['129:95'].inputs.unet_name, 'hi.safetensors');
  assert.equal(copy['129:96'].inputs.unet_name, 'lo.safetensors');
  assert.equal(copy['129:84'].inputs.clip_name, 'umt5.safetensors');
  assert.equal(copy['129:90'].inputs.vae_name, 'wanvae.safetensors');
  assert.equal(copy['129:101'].inputs.lora_name, 'lora-hi.safetensors');
  assert.equal(copy['129:102'].inputs.lora_name, 'lora-lo.safetensors');
  // nothing but those six inputs changed - in particular the Fast/High switch and the samplers
  const changed = Object.keys(copy).flatMap((id) => Object.keys(copy[id].inputs).filter((k) => JSON.stringify(copy[id].inputs[k]) !== JSON.stringify((JSON.parse(before) as Template)[id].inputs[k])));
  assert.equal(changed.length, 6);
});

test('the built-in video profile applied to the Wan template changes nothing', () => {
  const family = PROFILE_FAMILIES.find((f) => f.family === 'wan22-i2v')!;
  const copy = JSON.parse(JSON.stringify(wanTemplate)) as Template;
  applyModelSettings(copy, 'wan22-i2v', profileSettings({ family: 'wan22-i2v', files: Object.fromEntries(family.slots.map((s) => [s.key, s.defaultFile])), sampler: { steps: 1, cfg: 0, sampler: '', scheduler: '', shift: 0 } }));
  assert.deepEqual(copy, wanTemplate);
});
