import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SOURCE_IMAGE_FAMILIES, needsSourceImage } from './sourceFamilies';

test('videos, image to image, inpainting and picture upscales are made from a supplied picture; nothing else is', () => {
  assert.ok(needsSourceImage('wan22-i2v'));
  assert.ok(needsSourceImage('z-image-i2i'));
  assert.ok(needsSourceImage('z-image-inpaint'));
  assert.ok(needsSourceImage('upscale-image'));
  assert.ok(!needsSourceImage('z-image'));
  assert.ok(!needsSourceImage('wan22-t2v'));
  // A video upscale works from a library video, which the Library itself keeps.
  assert.ok(!needsSourceImage('upscale-video'));
  assert.ok(!needsSourceImage('something-new'));
  assert.equal(SOURCE_IMAGE_FAMILIES.size, 4);
});
