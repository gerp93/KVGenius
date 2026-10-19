import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SOURCE_IMAGE_FAMILIES, needsSourceImage } from './sourceFamilies';

test('videos, image to image, inpainting and picture upscales are made from a supplied picture; nothing else is', () => {
  assert.ok(needsSourceImage('wan22-i2v'));
  assert.ok(needsSourceImage('z-image-i2i'));
  assert.ok(needsSourceImage('z-image-inpaint'));
  assert.ok(needsSourceImage('z-image-outpaint'));
  assert.ok(needsSourceImage('upscale-image'));
  // A face reference is drawn from the reference picture, so its result keeps a copy of it.
  assert.ok(needsSourceImage('face-klein'));
  assert.ok(needsSourceImage('face-qwen'));
  assert.ok(!needsSourceImage('z-image'));
  assert.ok(!needsSourceImage('wan22-t2v'));
  // A video upscale works from a library video, which the Library itself keeps.
  assert.ok(!needsSourceImage('upscale-video'));
  assert.ok(!needsSourceImage('something-new'));
  assert.equal(SOURCE_IMAGE_FAMILIES.size, 7);
});
