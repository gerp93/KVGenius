import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_DENOISE, I2I_FAMILY, clampDenoise, imageFamilyFor } from './imageToImage';
import { needsSourceImage } from './sourceFamilies';
import { FAMILY_KIND } from './types';
import { generationOrigin, FAMILIES_BY_ORIGIN, originsForKind, passesOriginFilter } from './origin';

test('strength stays within what the sampler accepts, and junk falls back to the default', () => {
  assert.equal(clampDenoise(0.5), 0.5);
  assert.equal(clampDenoise(0), 0.05);
  assert.equal(clampDenoise(7), 1);
  assert.equal(clampDenoise('0.35'), 0.35);
  assert.equal(clampDenoise(0.123456), 0.12);
  assert.equal(clampDenoise(undefined), DEFAULT_DENOISE);
  assert.equal(clampDenoise('nope'), DEFAULT_DENOISE);
  assert.equal(clampDenoise(NaN), DEFAULT_DENOISE);
});

test('a start picture switches the family, otherwise the text family is used', () => {
  assert.equal(imageFamilyFor('z-image', true), I2I_FAMILY);
  assert.equal(imageFamilyFor('z-image', false), 'z-image');
});

test('image to image is made from a supplied picture, produces an image, and has its own origin tag', () => {
  assert.equal(needsSourceImage(I2I_FAMILY), true);
  assert.equal(FAMILY_KIND[I2I_FAMILY], 'image');
  assert.equal(generationOrigin(I2I_FAMILY)?.kind, 'image-to-image');
  assert.equal(generationOrigin(I2I_FAMILY)?.label, 'Image → Image');
  assert.deepEqual(FAMILIES_BY_ORIGIN['image-to-image'], [I2I_FAMILY]);
  assert.equal(passesOriginFilter(I2I_FAMILY, ['image-to-image']), true);
  assert.equal(passesOriginFilter(I2I_FAMILY, ['text-to-image']), false);
  assert.equal(passesOriginFilter('z-image', ['image-to-image']), false);
  assert.ok(originsForKind('image').includes('image-to-image'));
  assert.ok(!originsForKind('video').includes('image-to-image'));
});
