import { test } from 'node:test';
import assert from 'node:assert/strict';
import { importSlot, UPSCALE_IMPORT_FAMILY } from './modelFamilies';

test('a model file can be imported for an upscale model as well as for a profile slot', () => {
  const slot = importSlot(UPSCALE_IMPORT_FAMILY, 'model');
  assert.equal(slot?.folder, 'upscale_models');
  assert.equal(importSlot(UPSCALE_IMPORT_FAMILY, 'other'), undefined);
  assert.equal(importSlot('z-image', 'vae')?.folder, 'vae');
  assert.equal(importSlot('nonsense', 'vae'), undefined);
  assert.equal(importSlot(42, 'vae'), undefined);
});
