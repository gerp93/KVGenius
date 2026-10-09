import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PromptStyle, cleanStyleKind, combinePrompt, extraWording, styleLabel, validateStyleInput } from './styles';

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
  assert.deepEqual(validateStyleInput({ name: '  Poster ', text: ' bold  ' }), { ok: true, value: { name: 'Poster', text: 'bold', kind: 'style' } });
  assert.equal(validateStyleInput({ name: ' ', text: 'x' }).ok, false);
  assert.equal(validateStyleInput({ name: 'x', text: '  ' }).ok, false);
  assert.equal(validateStyleInput({ name: 'x'.repeat(61), text: 'x' }).ok, false);
  assert.equal(validateStyleInput({ name: 'x', text: 'x'.repeat(2001) }).ok, false);
});

const piece = (id: number, name: string, text: string, kind: 'style' | 'element'): PromptStyle => ({ id, name, text, kind, createdAt: '' });

test('a kind defaults to style and anything unknown is a style', () => {
  assert.equal(cleanStyleKind('element'), 'element');
  assert.equal(cleanStyleKind('style'), 'style');
  assert.equal(cleanStyleKind('nope'), 'style');
  assert.equal(cleanStyleKind(undefined), 'style');
  const element = validateStyleInput({ name: 'Coat', text: 'red trench coat', kind: 'element' });
  assert.deepEqual(element, { ok: true, value: { name: 'Coat', text: 'red trench coat', kind: 'element' } });
});

test('a list of pieces is added in order, skipping blanks', () => {
  assert.equal(combinePrompt('a fox', ['red coat', '', null, 'bold lithograph']), 'a fox, red coat, bold lithograph');
  assert.equal(combinePrompt('A fox.', ['red coat', 'poster']), 'A fox. red coat, poster');
  assert.equal(combinePrompt('', ['red coat', 'poster']), 'red coat, poster');
  assert.equal(combinePrompt('a fox', []), 'a fox');
});

test('elements go before the style, and the label names both', () => {
  const coat = piece(1, 'Coat', 'red trench coat', 'element');
  const hat = piece(2, 'Hat', 'wide-brim hat', 'element');
  const anime = piece(3, 'Anime', 'cel shaded', 'style');
  assert.deepEqual(extraWording([coat, hat], anime), ['red trench coat', 'wide-brim hat', 'cel shaded']);
  assert.equal(combinePrompt('a fox', extraWording([coat, hat], anime)), 'a fox, red trench coat, wide-brim hat, cel shaded');
  assert.equal(styleLabel([coat, hat], anime), 'Anime + Coat + Hat');
  assert.equal(styleLabel([coat], null), 'Coat');
  assert.equal(styleLabel([], null), null);
  assert.equal(combinePrompt('a fox', extraWording([], null)), 'a fox');
});
