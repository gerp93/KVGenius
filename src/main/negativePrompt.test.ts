import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fillTextToImage } from './textToImagePatch';
import { findDuplicateGeneration, getGenerationById, initDatabase, insertGeneration } from './db';
import { DEFAULT_NEGATIVE_PROMPTS, NEGATIVE_PROMPT_NODE, defaultNegativePrompt, negativePromptForJob, usesNegativePrompt } from '../shared/negativePrompt';
import { defaultSlotData } from '../shared/promptSlots';
import type { GenerationParams } from '../shared/types';
import zImage from './templates/z-image.json';
import qwen from './templates/qwen-image.json';
import flux from './templates/flux-dev.json';
import hiDream from './templates/hidream.json';
import ovis from './templates/ovis-image.json';
import chroma from './templates/chroma.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const templates: Record<string, Template> = {
  'z-image': zImage as unknown as Template,
  'qwen-image': qwen as unknown as Template,
  'flux-dev': flux as unknown as Template,
  hidream: hiDream as unknown as Template,
  'ovis-image': ovis as unknown as Template,
  chroma: chroma as unknown as Template,
};
const base: GenerationParams = { prompt: 'a red fox', width: 832, height: 1216, seed: 7, steps: 12, cfg: 2.5 };
const run = (family: string, extra: Partial<GenerationParams> = {}): Template => {
  const copy = JSON.parse(JSON.stringify(templates[family])) as Template;
  fillTextToImage(copy, family, { ...base, ...extra });
  return copy;
};

test('exactly the families whose sampler uses its negative prompt offer one, and their defaults are what the templates ship', () => {
  assert.deepEqual(Object.keys(DEFAULT_NEGATIVE_PROMPTS).sort(), ['chroma', 'hidream', 'ovis-image', 'qwen-image']);
  for (const [family, template] of Object.entries(templates)) {
    const sampler = Object.values(template).find((n) => n.class_type === 'KSampler')!;
    const negativeNode = template[(sampler.inputs.negative as [string, number])[0]];
    if (usesNegativePrompt(family)) {
      assert.equal(negativeNode.class_type, 'CLIPTextEncode', family);
      assert.equal(template[NEGATIVE_PROMPT_NODE], negativeNode, `${family}: the negative prompt is in node ${NEGATIVE_PROMPT_NODE}`);
      assert.equal(negativeNode.inputs.text, DEFAULT_NEGATIVE_PROMPTS[family], `${family}: the default is the template's own`);
    } else {
      assert.equal(negativeNode.class_type, 'ConditioningZeroOut', `${family}: its negative is zeroed out, so no field`);
    }
  }
  assert.equal(defaultNegativePrompt('flux-dev'), '');
});

test('a negative prompt is written into its node only for those families, and only when given', () => {
  for (const family of ['qwen-image', 'hidream', 'ovis-image', 'chroma']) {
    assert.equal(run(family, { negativePrompt: 'blurry, extra fingers' })[NEGATIVE_PROMPT_NODE].inputs.text, 'blurry, extra fingers', family);
    assert.equal(run(family, { negativePrompt: '' })[NEGATIVE_PROMPT_NODE].inputs.text, '', `${family}: an empty text is a real choice`);
    assert.equal(run(family)[NEGATIVE_PROMPT_NODE].inputs.text, DEFAULT_NEGATIVE_PROMPTS[family], `${family}: absent keeps the template's own`);
  }
  // Z-Image and Flux have no negative prompt to set: the graph is left as it is.
  for (const family of ['z-image', 'flux-dev']) {
    assert.deepEqual(run(family, { negativePrompt: 'blurry' }), run(family), family);
  }
});

test('only text that differs from the default is sent, and nothing for a family that takes none', () => {
  assert.equal(negativePromptForJob('chroma', null), undefined);
  assert.equal(negativePromptForJob('chroma', DEFAULT_NEGATIVE_PROMPTS.chroma), undefined);
  assert.equal(negativePromptForJob('chroma', 'blurry'), 'blurry');
  assert.equal(negativePromptForJob('chroma', ''), '');
  assert.equal(negativePromptForJob('hidream', 'x'.repeat(5000))?.length, 2000);
  assert.equal(negativePromptForJob('flux-dev', 'blurry'), undefined);
  assert.equal(negativePromptForJob('z-image', 'blurry'), undefined);
});

test('a generation keeps the negative prompt it was made with, and a different one is a different picture', () => {
  const db = initDatabase(':memory:');
  const typed = insertGeneration(db, { ...base, negativePrompt: 'blurry' }, 'chroma', '/out/a.png');
  assert.equal(typed.negativePrompt, 'blurry');
  assert.equal(getGenerationById(db, typed.id)?.negativePrompt, 'blurry');
  const none = insertGeneration(db, { ...base, negativePrompt: '' }, 'chroma', '/out/b.png');
  assert.equal(getGenerationById(db, none.id)?.negativePrompt, '', 'nothing to avoid is kept as nothing, not as the default');
  const plain = insertGeneration(db, base, 'chroma', '/out/c.png');
  assert.equal(getGenerationById(db, plain.id)?.negativePrompt, null, 'the default is stored as no text');

  const query = { workspaceId: 1, prompt: 'a red fox', width: 832, height: 1216, seed: 7, steps: 12, cfg: 2.5 };
  assert.equal(findDuplicateGeneration(db, 'chroma', { ...query, negativePrompt: 'blurry' })?.id, typed.id);
  assert.equal(findDuplicateGeneration(db, 'chroma', { ...query, negativePrompt: '' })?.id, none.id);
  assert.equal(findDuplicateGeneration(db, 'chroma', query)?.id, plain.id);
  assert.equal(findDuplicateGeneration(db, 'chroma', { ...query, negativePrompt: 'other' }), null);
});

test('a Generate tab starts on the default negative prompt', () => {
  assert.equal(defaultSlotData('image').negativePrompt, null);
});
