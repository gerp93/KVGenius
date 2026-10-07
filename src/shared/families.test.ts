import { test } from 'node:test';
import assert from 'node:assert/strict';
import { canonicalFamily, LEGACY_FAMILY_KEYS, Z_IMAGE_FAMILY } from './families';
import { generationOrigin, passesOriginFilter } from './origin';

test('the retired key maps to the current one; everything else is untouched', () => {
  assert.equal(canonicalFamily('z-image-turbo'), Z_IMAGE_FAMILY);
  assert.equal(canonicalFamily('z-image'), 'z-image');
  assert.equal(canonicalFamily('wan22-i2v'), 'wan22-i2v');
  assert.equal(canonicalFamily('something-new'), 'something-new');
});

test('a row the migration has not reached yet still behaves like its family', () => {
  for (const [legacy, current] of Object.entries(LEGACY_FAMILY_KEYS)) {
    assert.deepEqual(generationOrigin(legacy), generationOrigin(current));
    assert.equal(passesOriginFilter(legacy, ['text-to-image']), true);
  }
});
