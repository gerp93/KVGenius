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

import { INPAINT_FAMILY, isPictureStartFamily } from './imageToImage';
import { profileFamilyKey } from './modelFamilies';

test('a mask on a source image makes it inpainting; a mask alone means nothing', () => {
  assert.equal(imageFamilyFor('z-image', true, true), INPAINT_FAMILY);
  assert.equal(imageFamilyFor('z-image', true, false), I2I_FAMILY);
  assert.equal(imageFamilyFor('z-image', false, true), 'z-image');
  assert.equal(isPictureStartFamily(INPAINT_FAMILY), true);
  assert.equal(isPictureStartFamily(I2I_FAMILY), true);
  assert.equal(isPictureStartFamily('z-image'), false);
});

test('inpainting is made from a supplied picture, makes an image, has its own tag and uses Z-Image\'s profiles', () => {
  assert.equal(needsSourceImage(INPAINT_FAMILY), true);
  assert.equal(FAMILY_KIND[INPAINT_FAMILY], 'image');
  assert.equal(generationOrigin(INPAINT_FAMILY)?.kind, 'inpaint');
  assert.equal(generationOrigin(INPAINT_FAMILY)?.label, 'Inpainted');
  assert.deepEqual(FAMILIES_BY_ORIGIN.inpaint, [INPAINT_FAMILY]);
  assert.equal(passesOriginFilter(INPAINT_FAMILY, ['inpaint']), true);
  assert.equal(passesOriginFilter(I2I_FAMILY, ['inpaint']), false);
  assert.equal(profileFamilyKey(INPAINT_FAMILY), 'z-image');
  assert.ok(originsForKind('image').includes('inpaint'));
  assert.ok(!originsForKind('video').includes('inpaint'));
});

import { OUTPAINT_FAMILY, OUTPAINT_MAX_PAD, normalizeOutpaint, outpaintOutputSize, parseOutpaint, serializeOutpaint } from './imageToImage';

test('an extension makes it outpainting, and wins over a mask; without a source image it means nothing', () => {
  assert.equal(imageFamilyFor('z-image', true, false, true), OUTPAINT_FAMILY);
  assert.equal(imageFamilyFor('z-image', true, true, true), OUTPAINT_FAMILY);
  assert.equal(imageFamilyFor('z-image', false, false, true), 'z-image');
  assert.equal(isPictureStartFamily(OUTPAINT_FAMILY), true);
});

test('outpainting is made from a supplied picture, makes an image, has its own tag and uses Z-Image\'s profiles', () => {
  assert.equal(needsSourceImage(OUTPAINT_FAMILY), true);
  assert.equal(FAMILY_KIND[OUTPAINT_FAMILY], 'image');
  assert.equal(generationOrigin(OUTPAINT_FAMILY)?.kind, 'outpaint');
  assert.deepEqual(FAMILIES_BY_ORIGIN.outpaint, [OUTPAINT_FAMILY]);
  assert.equal(profileFamilyKey(OUTPAINT_FAMILY), 'z-image');
  assert.ok(originsForKind('image').includes('outpaint'));
  assert.ok(!originsForKind('video').includes('outpaint'));
});

test('a padding is whole pixels within limits, and none at all is null', () => {
  assert.equal(normalizeOutpaint(null), null);
  assert.equal(normalizeOutpaint({ left: 0, top: 0, right: 0, bottom: 0 }), null);
  assert.equal(normalizeOutpaint({ left: -5, right: 'x' }), null);
  assert.deepEqual(normalizeOutpaint({ left: 100.4, top: 99999, right: 0, bottom: '64' }), { left: 100, top: OUTPAINT_MAX_PAD, right: 0, bottom: 64 });
});

test('a padding round-trips through its stored string', () => {
  const pad = { left: 256, top: 0, right: 128, bottom: 64 };
  assert.equal(serializeOutpaint(pad), '256,0,128,64');
  assert.deepEqual(parseOutpaint(serializeOutpaint(pad)), pad);
  assert.equal(serializeOutpaint(null), null);
  assert.equal(parseOutpaint(null), null);
  assert.equal(parseOutpaint('nonsense'), null);
});

test('the extended picture is drawn at its whole canvas, long side kept within 1024-1536, sides in 64s', () => {
  assert.deepEqual(outpaintOutputSize(1024, 1024, { left: 512, top: 0, right: 0, bottom: 0 }), { width: 1536, height: 1024 });
  // a small picture is drawn larger rather than tiny
  assert.deepEqual(outpaintOutputSize(512, 512, { left: 0, top: 0, right: 512, bottom: 0 }), { width: 1024, height: 512 });
  // a big one is brought down
  const big = outpaintOutputSize(4000, 3000, { left: 1000, top: 0, right: 1000, bottom: 0 });
  assert.equal(Math.max(big.width, big.height), 1536);
  assert.equal(big.width % 64, 0);
  assert.equal(big.height % 64, 0);
});
