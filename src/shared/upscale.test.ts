import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_UPSCALE_FACTOR, fileNameOf, isUpscaleFamily, nearestUpscaleFactor, upscaledSize, UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY } from './upscale';

test('upscaledSize rounds a picture to whole pixels', () => {
  assert.deepEqual(upscaledSize(1000, 750, 2), { width: 2000, height: 1500 });
  assert.deepEqual(upscaledSize(333, 333, 1.5), { width: 500, height: 500 });
});

test('upscaledSize keeps a video on even sides', () => {
  const { width, height } = upscaledSize(333, 251, 1.5, true);
  assert.equal(width % 2, 0);
  assert.equal(height % 2, 0);
  assert.deepEqual(upscaledSize(640, 480, 2, true), { width: 1280, height: 960 });
});

test('fileNameOf handles both path separators', () => {
  assert.equal(fileNameOf(String.raw`C:\Users\me\pics\cat.png`), 'cat.png');
  assert.equal(fileNameOf('/home/me/pics/cat.png'), 'cat.png');
  assert.equal(fileNameOf('cat.png'), 'cat.png');
});

test('isUpscaleFamily knows both upscale families and nothing else', () => {
  assert.ok(isUpscaleFamily(UPSCALE_FAMILY));
  assert.ok(isUpscaleFamily(UPSCALE_VIDEO_FAMILY));
  assert.ok(!isUpscaleFamily('z-image-turbo'));
});

test('nearestUpscaleFactor picks the offered multiplier closest to the real one', () => {
  assert.equal(nearestUpscaleFactor(1000, 2000), 2);
  assert.equal(nearestUpscaleFactor(1000, 1500), 1.5);
  assert.equal(nearestUpscaleFactor(1000, 4000), 4);
  assert.equal(nearestUpscaleFactor(1000, 2900), 3);
  assert.equal(nearestUpscaleFactor(0, 2000), DEFAULT_UPSCALE_FACTOR);
});
