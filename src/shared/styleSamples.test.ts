import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BASELINE_ID, DEFAULT_SAMPLE_SETTINGS, StyleSampleInfo, idsToRender, sampleFreshness, validateSampleSettings } from './styleSamples';

const settings = { prompt: 'a fox', seed: 7 };

test('an example is current only while its wording, the standard prompt and the seed are as they were', () => {
  const snap = { text: 'bold lithograph', prompt: 'a fox', seed: 7 };
  assert.equal(sampleFreshness(null, 'bold lithograph', settings), 'none');
  assert.equal(sampleFreshness(snap, 'bold lithograph', settings), 'current');
  assert.equal(sampleFreshness(snap, 'bold lithograph, limited palette', settings), 'outdated');
  assert.equal(sampleFreshness(snap, 'bold lithograph', { ...settings, prompt: 'a cat' }), 'outdated');
  assert.equal(sampleFreshness(snap, 'bold lithograph', { ...settings, seed: 8 }), 'outdated');
});

test('settings are trimmed and checked', () => {
  assert.deepEqual(validateSampleSettings({ prompt: '  a fox ', seed: '12' }), { ok: true, value: { prompt: 'a fox', seed: 12 } });
  assert.equal(validateSampleSettings({ prompt: ' ', seed: 1 }).ok, false);
  assert.equal(validateSampleSettings({ prompt: 'x', seed: -1 }).ok, false);
  assert.equal(validateSampleSettings({ prompt: 'x', seed: 1.5 }).ok, false);
  assert.equal(validateSampleSettings({ prompt: 'x', seed: 2 ** 32 }).ok, false);
  assert.equal(validateSampleSettings({ prompt: 'x'.repeat(1001), seed: 1 }).ok, false);
  assert.equal(validateSampleSettings(DEFAULT_SAMPLE_SETTINGS).ok, true);
});

test('what to render: everything not current and not already in the queue, or everything with `all`', () => {
  const info = (id: number, state: StyleSampleInfo['state']): StyleSampleInfo => ({ id, state, imageUrl: null, renderedAt: null, error: null });
  const samples = [info(BASELINE_ID, 'current'), info(1, 'outdated'), info(2, 'none'), info(3, 'queued'), info(4, 'failed'), info(5, 'current'), info(6, 'rendering')];
  assert.deepEqual(idsToRender(samples, false), [1, 2, 4]);
  assert.deepEqual(idsToRender(samples, true), [0, 1, 2, 4, 5]);
});
