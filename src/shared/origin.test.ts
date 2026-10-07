import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GIF_FAMILY } from './gif';
import { generationOrigin } from './origin';
import { UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY } from './upscale';

test('each known family has an origin', () => {
  assert.equal(generationOrigin('z-image-turbo')?.kind, 'text-to-image');
  assert.equal(generationOrigin('wan22-i2v')?.kind, 'image-to-video');
  assert.equal(generationOrigin(GIF_FAMILY)?.kind, 'gif');
});

test('image and video upscales are both upscales', () => {
  assert.equal(generationOrigin(UPSCALE_FAMILY)?.kind, 'upscale');
  assert.equal(generationOrigin(UPSCALE_VIDEO_FAMILY)?.kind, 'upscale');
});

test('an unknown family has no origin rather than a wrong one', () => {
  assert.equal(generationOrigin('some-future-model'), null);
});
