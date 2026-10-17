import assert from 'node:assert/strict';
import test from 'node:test';
import { joinSections, validateTemplateInput, valuesFromPrompt } from './promptTemplates';

const sections = [
  { id: 'a', label: 'Subject', hint: '' },
  { id: 'b', label: 'Outfit', hint: '' },
  { id: 'c', label: 'Camera', hint: '' },
];

test('joinSections joins the filled boxes in order, skipping blank ones', () => {
  const template = { sections, separator: 'comma' as const };
  assert.equal(joinSections(template, { a: 'a woman', c: ' 35mm lens ' }), 'a woman, 35mm lens');
  assert.equal(joinSections(template, {}), '');
});

test('joinSections uses a plain space after punctuation, like styles do', () => {
  const template = { sections, separator: 'comma' as const };
  assert.equal(joinSections(template, { a: 'A woman.', b: 'red dress' }), 'A woman. red dress');
});

test('joinSections honours the space and new line separators', () => {
  assert.equal(joinSections({ sections, separator: 'space' }, { a: 'x', b: 'y' }), 'x y');
  assert.equal(joinSections({ sections, separator: 'newline' }, { a: 'x', c: 'y' }), 'x\ny');
});

test('valuesFromPrompt puts the existing prompt in the first section', () => {
  assert.deepEqual(valuesFromPrompt({ sections }, 'a cat'), { a: 'a cat' });
  assert.deepEqual(valuesFromPrompt({ sections }, '   '), {});
});

test('validateTemplateInput trims, keeps ids and fills in missing or repeated ones', () => {
  const result = validateTemplateInput({
    name: '  Portrait ',
    sections: [{ id: 'x', label: ' Subject ', hint: ' who ' }, { id: 'x', label: 'Outfit' }, { label: 'Camera' }],
    separator: 'space',
  });
  assert.ok(result.ok);
  if (!result.ok) return;
  assert.equal(result.value.name, 'Portrait');
  assert.equal(result.value.separator, 'space');
  assert.deepEqual(result.value.sections.map((s) => s.label), ['Subject', 'Outfit', 'Camera']);
  assert.equal(result.value.sections[0].id, 'x');
  assert.equal(result.value.sections[0].hint, 'who');
  assert.equal(new Set(result.value.sections.map((s) => s.id)).size, 3);
});

test('validateTemplateInput rejects a missing name, no sections and a nameless section', () => {
  assert.equal(validateTemplateInput({ name: ' ', sections: [{ label: 'a' }] }).ok, false);
  assert.equal(validateTemplateInput({ name: 'T', sections: [] }).ok, false);
  assert.equal(validateTemplateInput({ name: 'T', sections: [{ label: ' ' }] }).ok, false);
});
