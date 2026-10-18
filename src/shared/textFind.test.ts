import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findMatches, matchNearCaret, stepMatch } from './textFind';

test('finds every occurrence, ignoring case', () => {
  assert.deepEqual(findMatches('A red fox, a RED hat', 'red'), [
    { start: 2, end: 5 },
    { start: 13, end: 16 },
  ]);
});

test('an empty query or no occurrence finds nothing', () => {
  assert.deepEqual(findMatches('a red fox', ''), []);
  assert.deepEqual(findMatches('a red fox', 'blue'), []);
});

test('matches do not overlap', () => {
  assert.deepEqual(findMatches('aaa', 'aa'), [{ start: 0, end: 2 }]);
});

test('text whose lower-case form changes length still gets sensible positions', () => {
  // "İ" lower-cases to two characters, which would shift every later position.
  const matches = findMatches('İ fox', 'fox');
  assert.deepEqual(matches, [{ start: 2, end: 5 }]);
});

test('stepping wraps at both ends', () => {
  assert.equal(stepMatch(-1, 3, 1), 0);
  assert.equal(stepMatch(-1, 3, -1), 2);
  assert.equal(stepMatch(2, 3, 1), 0);
  assert.equal(stepMatch(0, 3, -1), 2);
  assert.equal(stepMatch(1, 3, 1), 2);
  assert.equal(stepMatch(0, 0, 1), -1);
});

test('a search starts from the caret', () => {
  const matches = findMatches('red fox red hat red', 'red');
  assert.equal(matchNearCaret(matches, 0), 0);
  assert.equal(matchNearCaret(matches, 4), 1);
  assert.equal(matchNearCaret(matches, 17), 2);
  assert.equal(matchNearCaret(matches, 19), 0);
  assert.equal(matchNearCaret([], 5), -1);
});
