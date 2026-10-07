import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GIF_FAMILY } from './gif';
import { originsForKind, cleanOrigins, FAMILIES_BY_ORIGIN, generationOrigin, ORIGIN_KINDS, passesOriginFilter } from './origin';
import { UPSCALE_FAMILY, UPSCALE_VIDEO_FAMILY } from './upscale';

test('each known family has an origin', () => {
  assert.equal(generationOrigin('z-image')?.kind, 'text-to-image');
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

test('every family listed for an origin maps back to that origin', () => {
  for (const kind of ORIGIN_KINDS) {
    for (const family of FAMILIES_BY_ORIGIN[kind]) assert.equal(generationOrigin(family)?.kind, kind);
  }
});

test('cleanOrigins keeps real origins only, in a fixed order, without repeats', () => {
  assert.deepEqual(cleanOrigins(['gif', 'nonsense', 'upscale', 'gif']), ['upscale', 'gif']);
  assert.deepEqual(cleanOrigins('upscale'), []);
  assert.deepEqual(cleanOrigins(undefined), []);
});

test('an origin filter passes the chosen origins; none chosen passes everything', () => {
  assert.equal(passesOriginFilter('upscale-image', []), true);
  assert.equal(passesOriginFilter('upscale-image', ['upscale']), true);
  assert.equal(passesOriginFilter('upscale-video', ['upscale']), true);
  assert.equal(passesOriginFilter('z-image', ['upscale']), false);
  assert.equal(passesOriginFilter('some-future-model', ['upscale']), false);
});

test('each tab offers only the origins that can occur on it', () => {
  assert.deepEqual(originsForKind('image'), ['text-to-image', 'image-to-image', 'upscale', 'gif']);
  assert.deepEqual(originsForKind('video'), ['image-to-video', 'upscale']);
});
