import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isDistilledName, suggestSettings } from './settingsSuggestion';

test('a file named like a distilled model gets the built-in model settings', () => {
  const s = suggestSettings('z-image', { diffusionModel: 'z_image_turbo_bf16.safetensors' });
  assert.deepEqual(s?.settings, { steps: 8, cfg: 1, sampler: 'res_multistep', scheduler: 'simple', shift: 3 });
  assert.match(s?.reason ?? '', /distilled/);
  assert.equal(isDistilledName('foo_LIGHTNING_v2.safetensors'), true);
  assert.equal(isDistilledName('capybara_v0.1.safetensors'), false);
});

test('any other file gets more steps and real guidance, keeping the family sampler, scheduler and shift', () => {
  const s = suggestSettings('z-image', { diffusionModel: 'z_image_base_bf16.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' });
  assert.deepEqual(s?.settings, { steps: 30, cfg: 4, sampler: 'res_multistep', scheduler: 'simple', shift: 3 });
  assert.match(s?.reason ?? '', /not say it is distilled/);
});

test('the same files always give the same answer, and the other files do not change it', () => {
  const a = suggestSettings('z-image', { diffusionModel: 'x.safetensors', vae: 'one.safetensors' });
  const b = suggestSettings('z-image', { diffusionModel: 'x.safetensors', vae: 'two.safetensors' });
  assert.deepEqual(a, b);
});

test('no model picked yet starts from the built-in; families with no driven sampler get nothing', () => {
  assert.equal(suggestSettings('z-image', {})?.settings.steps, 8);
  assert.equal(suggestSettings('wan22-i2v', { highNoiseModel: 'x.safetensors' }), null);
  assert.equal(suggestSettings('nonexistent', {}), null);
});

test('the newer families start from the full model, and only a distilled-looking name changes that', () => {
  assert.deepEqual(suggestSettings('qwen-image', { diffusionModel: 'qwen_image_2512_fp8_e4m3fn.safetensors' })?.settings, { steps: 50, cfg: 4, sampler: 'euler', scheduler: 'simple', shift: 3.1 });
  assert.deepEqual(suggestSettings('flux-dev', { diffusionModel: 'flux1-dev.safetensors' })?.settings, { steps: 20, cfg: 3.5, sampler: 'euler', scheduler: 'simple', shift: 0 });
  assert.deepEqual(suggestSettings('hidream', { diffusionModel: 'hidream_i1_full_fp8.safetensors' })?.settings, { steps: 50, cfg: 5, sampler: 'uni_pc', scheduler: 'simple', shift: 3 });
  assert.deepEqual(suggestSettings('ovis-image', { diffusionModel: 'ovis_image_bf16.safetensors' })?.settings, { steps: 20, cfg: 5, sampler: 'euler', scheduler: 'simple', shift: 3 });
  assert.match(suggestSettings('flux-dev', { diffusionModel: 'flux1-dev.safetensors' })?.reason ?? '', /ComfyUI/);
});

test('a distilled Qwen-Image or Flux file gets a few steps, read from its name when it says', () => {
  const lightning = suggestSettings('qwen-image', { diffusionModel: 'qwen_image_Lightning-4steps.safetensors' });
  assert.equal(lightning?.settings.steps, 4);
  assert.equal(lightning?.settings.cfg, 1);
  assert.equal(suggestSettings('qwen-image', { diffusionModel: 'qwen_image_turbo.safetensors' })?.settings.steps, 8);
  // Flux keeps its guidance (it is not CFG), only the steps shrink: schnell is a 4-step model, another distilled name a few steps
  const schnell = suggestSettings('flux-dev', { diffusionModel: 'flux1-schnell.safetensors' });
  assert.equal(schnell?.settings.steps, 4);
  assert.equal(schnell?.settings.cfg, 3.5);
  assert.equal(suggestSettings('flux-dev', { diffusionModel: 'flux_hyper_8steps.safetensors' })?.settings.steps, 8);
});

test('HiDream Dev and Fast have their own sampler, scheduler, steps and shift', () => {
  assert.deepEqual(suggestSettings('hidream', { diffusionModel: 'hidream_i1_fast_fp8.safetensors' })?.settings, { steps: 16, cfg: 1, sampler: 'lcm', scheduler: 'normal', shift: 3 });
  assert.deepEqual(suggestSettings('hidream', { diffusionModel: 'hidream_i1_dev_fp8.safetensors' })?.settings, { steps: 28, cfg: 1, sampler: 'lcm', scheduler: 'normal', shift: 6 });
});
