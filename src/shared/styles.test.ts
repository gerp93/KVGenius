import { test } from 'node:test';
import assert from 'node:assert/strict';
import { combinePrompt, validateStyleInput } from './styles';

test('without a style the prompt is returned exactly as typed', () => {
  for (const prompt of ['a red fox', '  a red fox  ', 'a red fox\n', '', 'trailing comma,']) {
    assert.equal(combinePrompt(prompt, null), prompt);
    assert.equal(combinePrompt(prompt, undefined), prompt);
    assert.equal(combinePrompt(prompt, ''), prompt);
    assert.equal(combinePrompt(prompt, '   \n'), prompt, 'a blank style adds nothing');
  }
});

test('a style is appended after the prompt with a comma', () => {
  assert.equal(combinePrompt('a fox in a snowy forest', '1930s movie poster'), 'a fox in a snowy forest, 1930s movie poster');
  assert.equal(combinePrompt('a fox  \n', '  1930s movie poster '), 'a fox, 1930s movie poster', 'whitespace at the joint is tidied');
});

test('a prompt that already ends in punctuation just gets a space', () => {
  assert.equal(combinePrompt('a fox,', 'bold lithograph'), 'a fox, bold lithograph');
  assert.equal(combinePrompt('A fox.', 'bold lithograph'), 'A fox. bold lithograph');
  assert.equal(combinePrompt('A fox!', 'bold lithograph'), 'A fox! bold lithograph');
});

test('a blank prompt with a style sends just the style', () => {
  assert.equal(combinePrompt('  ', 'bold lithograph'), 'bold lithograph');
});

test('style input is trimmed and checked', () => {
  assert.deepEqual(validateStyleInput({ name: '  Poster ', text: ' bold  ' }), { ok: true, value: { name: 'Poster', text: 'bold' } });
  assert.equal(validateStyleInput({ name: ' ', text: 'x' }).ok, false);
  assert.equal(validateStyleInput({ name: 'x', text: '  ' }).ok, false);
  assert.equal(validateStyleInput({ name: 'x'.repeat(61), text: 'x' }).ok, false);
  assert.equal(validateStyleInput({ name: 'x', text: 'x'.repeat(2001) }).ok, false);
});
