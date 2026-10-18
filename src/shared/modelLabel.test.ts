import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GIF_FAMILY } from './gif';
import { modelLabel } from './modelLabel';
import { UPSCALE_FAMILY } from './upscale';

test('a saved model variant names the model', () => {
  assert.equal(modelLabel({ modelFamily: 'z-image', modelName: 'My finetune' }), 'My finetune');
});

test('without a variant, the family\'s built-in model is named', () => {
  assert.equal(modelLabel({ modelFamily: 'z-image', modelName: null }), 'Z Image Turbo');
  assert.equal(modelLabel({ modelFamily: 'qwen-image', modelName: null }), 'Qwen-Image');
  assert.equal(modelLabel({ modelFamily: 'wan22-t2v', modelName: null }), 'Wan 2.2 text to video');
});

test('the retired Z-Image key and the families that borrow its files resolve to it', () => {
  assert.equal(modelLabel({ modelFamily: 'z-image-turbo', modelName: null }), 'Z Image Turbo');
  assert.equal(modelLabel({ modelFamily: 'z-image-outpaint', modelName: null }), 'Z Image Turbo');
});

test('upscales, GIFs and unknown families have no model tag', () => {
  assert.equal(modelLabel({ modelFamily: UPSCALE_FAMILY, modelName: null }), null);
  assert.equal(modelLabel({ modelFamily: GIF_FAMILY, modelName: null }), null);
  assert.equal(modelLabel({ modelFamily: 'some-future-model', modelName: null }), null);
});
