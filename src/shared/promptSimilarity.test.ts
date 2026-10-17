import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  SIMILARITY,
  cleanSimilarity,
  clusterPrompts,
  orderBySimilarity,
  promptWords,
  similarity,
  similarityMatrix,
  varyingWords,
} from './promptSimilarity';

const base = 'a woman in a red dress standing on a beach at sunset, soft light, film grain, detailed skin, 35mm photo';
const blue = base.replace('red', 'blue');
const twoWords = base.replace('red', 'green').replace('beach', 'pier');
const other = 'a castle on a hill in thick fog, moody, dark fantasy, painterly, wide shot';

test('words are lowercase, distinct, and ignore emphasis weights', () => {
  assert.deepEqual(promptWords('Red DRESS, red (dress:1.2), it\'s'), ['red', 'dress', "it's"]);
  assert.deepEqual(promptWords(''), []);
});

test('similarity is the share of words in common', () => {
  assert.equal(similarity(promptWords(base), promptWords(base)), 1);
  assert.equal(similarity(promptWords('a b'), promptWords('c d')), 0);
  assert.equal(similarity([], []), 1);
  // One word changed out of a long prompt stays well above the default threshold; a different prompt is far below it.
  assert.ok(similarity(promptWords(base), promptWords(blue)) > 0.85);
  assert.ok(similarity(promptWords(base), promptWords(other)) < 0.2);
});

test('clustering groups near-identical prompts and leaves different ones alone', () => {
  const prompts = [base, other, blue, twoWords];
  const clusters = clusterPrompts(similarityMatrix(prompts), SIMILARITY.default);
  assert.deepEqual(clusters, [[0, 2, 3], [1]]);
});

test('a stricter threshold splits a cluster', () => {
  const prompts = [base, blue, twoWords];
  const matrix = similarityMatrix(prompts);
  assert.deepEqual(clusterPrompts(matrix, 0.7), [[0, 1, 2]]);
  assert.deepEqual(clusterPrompts(matrix, 0.95), [[0], [1], [2]]);
});

test('a chain of small changes does not glue together prompts that are far apart', () => {
  // a~b and b~c are close, but a and c share little: average linkage keeps c out.
  const a = 'one two three four five six';
  const b = 'one two three four five seven';
  const c = 'one two three four eight seven';
  const d = 'one two nine ten eight seven';
  const clusters = clusterPrompts(similarityMatrix([a, b, c, d]), 0.6);
  assert.ok(!clusters.some((cluster) => cluster.includes(0) && cluster.includes(3)));
});

test('clustering nothing, or one prompt, is fine', () => {
  assert.deepEqual(clusterPrompts([], 0.75), []);
  assert.deepEqual(clusterPrompts(similarityMatrix(['only']), 0.75), [[0]]);
});

test('ordering by similarity puts the closest prompts side by side', () => {
  const prompts = [base, other, blue, twoWords];
  const matrix = similarityMatrix(prompts);
  const ordered = orderBySimilarity([0, 1, 2, 3], matrix);
  assert.equal(ordered[0], 0);
  assert.deepEqual(ordered.slice(0, 3).sort(), [0, 2, 3]);
  assert.equal(ordered[3], 1);
  assert.deepEqual(orderBySimilarity([], matrix), []);
});

test('varying words are the ones not every prompt uses', () => {
  assert.deepEqual(varyingWords([base, blue]).sort(), ['blue', 'red']);
  assert.deepEqual(varyingWords([base]), []);
});

test('the similarity setting is kept within its limits', () => {
  assert.equal(cleanSimilarity('nope'), SIMILARITY.default);
  assert.equal(cleanSimilarity(5), SIMILARITY.max);
  assert.equal(cleanSimilarity(0), SIMILARITY.min);
  assert.equal(cleanSimilarity(0.8), 0.8);
});
