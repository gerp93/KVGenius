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
