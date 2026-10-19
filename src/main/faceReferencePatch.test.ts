import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FACE_FAMILIES, FACE_KLEIN_FAMILY, FACE_QWEN_FAMILY, isFaceReferenceFamily } from '../shared/faceReference';
import { SOURCE_IMAGE_FAMILIES } from '../shared/sourceFamilies';
import { FAMILY_KIND } from '../shared/types';
import { generationOrigin } from '../shared/origin';
import { fillFaceReference, FACE_KLEIN_NODE_MAP, FACE_QWEN_NODE_MAP } from './faceReferencePatch';
import kleinTemplate from './templates/face-klein.json';
import qwenTemplate from './templates/face-qwen.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const fresh = (family: string) => JSON.parse(JSON.stringify(family === FACE_KLEIN_FAMILY ? kleinTemplate : qwenTemplate)) as Template;
const params = { prompt: 'the woman in the reference photo, in a cafe', width: 832, height: 1216, seed: 4242, steps: 4, cfg: 1 };

for (const family of FACE_FAMILIES) {
  test(`${family}: nothing in the template points at a node that does not exist`, () => {
    const t = fresh(family);
    for (const [id, node] of Object.entries(t)) {
      for (const value of Object.values(node.inputs)) {
        if (Array.isArray(value) && typeof value[0] === 'string') assert.ok(t[value[0]], `${id} refers to ${value[0]}`);
      }
    }
  });

  test(`${family}: is registered as an image family made from a supplied picture`, () => {
    assert.ok(isFaceReferenceFamily(family));
    assert.equal(FAMILY_KIND[family], 'image');
    assert.ok(SOURCE_IMAGE_FAMILIES.has(family));
    assert.equal(generationOrigin(family)?.kind, 'face-reference');
  });
}

test('Klein: fills the picture, prompt, seed and the requested size', () => {
  const t = fresh(FACE_KLEIN_FAMILY);
  fillFaceReference(t, FACE_KLEIN_FAMILY, params, 'ref_123.png');
  assert.equal(t[FACE_KLEIN_NODE_MAP.loadImage].inputs.image, 'ref_123.png');
  assert.equal(t[FACE_KLEIN_NODE_MAP.prompt].inputs.text, params.prompt);
  assert.equal(t[FACE_KLEIN_NODE_MAP.noise].inputs.noise_seed, 4242);
  for (const node of [FACE_KLEIN_NODE_MAP.latent, FACE_KLEIN_NODE_MAP.scheduler]) {
    assert.equal(t[node].inputs.width, 832);
    assert.equal(t[node].inputs.height, 1216);
  }
  // the distilled model's own settings are left as the template has them
  assert.equal(t[FACE_KLEIN_NODE_MAP.scheduler].inputs.steps, 4);
  assert.equal(t['fr-guider'].inputs.cfg, 1);
});

test('Klein: the reference reaches both the positive and the negative conditioning', () => {
  const t = fresh(FACE_KLEIN_FAMILY);
  assert.equal(t['fr-refpos'].class_type, 'ReferenceLatent');
  assert.equal(t['fr-refneg'].class_type, 'ReferenceLatent');
  assert.deepEqual(t['fr-refpos'].inputs.latent, ['fr-encode', 0]);
  assert.deepEqual(t['fr-refneg'].inputs.latent, ['fr-encode', 0]);
  assert.deepEqual(t['fr-guider'].inputs.positive, ['fr-refpos', 0]);
  assert.deepEqual(t['fr-guider'].inputs.negative, ['fr-refneg', 0]);
});

test('Qwen: fills the picture, prompt (positive only), seed and the requested size', () => {
  const t = fresh(FACE_QWEN_FAMILY);
  fillFaceReference(t, FACE_QWEN_FAMILY, params, 'ref_123.png');
  assert.equal(t[FACE_QWEN_NODE_MAP.loadImage].inputs.image, 'ref_123.png');
  assert.equal(t[FACE_QWEN_NODE_MAP.positive].inputs.prompt, params.prompt);
  assert.equal(t['fq-neg'].inputs.prompt, '', 'the negative stays empty');
  assert.equal(t[FACE_QWEN_NODE_MAP.sampler].inputs.seed, 4242);
  assert.equal(t[FACE_QWEN_NODE_MAP.latent].inputs.width, 832);
  assert.equal(t[FACE_QWEN_NODE_MAP.latent].inputs.height, 1216);
  assert.equal(t[FACE_QWEN_NODE_MAP.sampler].inputs.steps, 4, 'the Lightning LoRA path');
  assert.equal(t[FACE_QWEN_NODE_MAP.sampler].inputs.cfg, 1);
});

test('filling a template that is not a face-reference family is refused', () => {
  assert.throws(() => fillFaceReference(fresh(FACE_KLEIN_FAMILY), 'z-image', params, 'x.png'), /not a face-reference family/);
});
