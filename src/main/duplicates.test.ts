import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import {
  countPinnedWithPrompt,
  findDuplicateGeneration,
  initDatabase,
  insertGeneration,
  listGenerations,
  setGenerationPinned,
} from './db';
import { moveToTrash } from './trash';

const VIDEO_FAMILIES = ['wan22-i2v'];
const base = { prompt: 'a red fox', width: 1024, height: 1024, seed: 7, steps: 8, cfg: 1 };

function add(db: DatabaseSync, over: Partial<typeof base> & { length?: number } = {}, family = 'z-image', source: string | null = null) {
  const params = { ...base, ...over };
  return insertGeneration(db, params, family, `/out/${Math.random()}.png`, null, false, source);
}

test('the same prompt, size, seed, steps and cfg is a duplicate; any difference is not', () => {
  const db = initDatabase(':memory:');
  const made = add(db);
  assert.equal(findDuplicateGeneration(db, 'z-image', base)?.id, made.id);

  for (const change of [{ seed: 8 }, { width: 1280 }, { height: 896 }, { steps: 9 }, { cfg: 2 }, { prompt: 'a red fox!' }]) {
    assert.equal(findDuplicateGeneration(db, 'z-image', { ...base, ...change }), null, JSON.stringify(change));
  }
  assert.equal(findDuplicateGeneration(db, 'upscale-image', base), null, 'another model family is not the same output');
});

test('a prompt that differs only by surrounding whitespace is still a duplicate', () => {
  const db = initDatabase(':memory:');
  const made = add(db, { prompt: 'a red fox ' });
  assert.equal(findDuplicateGeneration(db, 'z-image', { ...base, prompt: '  a red fox' })?.id, made.id);
});

test('what is in the Trash does not count, and the newest match is the one reported', () => {
  const db = initDatabase(':memory:');
  const older = add(db);
  const newer = add(db);
  assert.equal(findDuplicateGeneration(db, 'z-image', base)?.id, newer.id);

  moveToTrash(db, [newer.id], fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-trash-')), { includeKept: true });
  assert.equal(findDuplicateGeneration(db, 'z-image', base)?.id, older.id);
});

test('a video is a duplicate only with the same length and the same source image', () => {
  const db = initDatabase(':memory:');
  const made = add(db, { length: 81 }, 'wan22-i2v', '/sources/a.png');
  const query = { ...base, length: 81, sourceImagePath: '/sources/a.png' };
  assert.equal(findDuplicateGeneration(db, 'wan22-i2v', query)?.id, made.id);
  assert.equal(findDuplicateGeneration(db, 'wan22-i2v', { ...query, length: 97 }), null);
  assert.equal(findDuplicateGeneration(db, 'wan22-i2v', { ...query, sourceImagePath: '/sources/b.png' }), null);
  assert.equal(findDuplicateGeneration(db, 'wan22-i2v', { ...query, sourceImagePath: null }), null, 'no source yet: nothing to compare');
});

test('pinned items with the same prompt form a group of that size', () => {
  const db = initDatabase(':memory:');
  const a = add(db, { seed: 1 });
  const b = add(db, { seed: 2 });
  const other = add(db, { seed: 3, prompt: 'a blue owl' });
  setGenerationPinned(db, a.id, true);
  assert.equal(countPinnedWithPrompt(db, 'a red fox'), 1);
  setGenerationPinned(db, b.id, true);
  setGenerationPinned(db, other.id, true);
  assert.equal(countPinnedWithPrompt(db, 'a red fox'), 2);
  setGenerationPinned(db, a.id, false);
  assert.equal(countPinnedWithPrompt(db, 'a red fox'), 1);
});

test('a stack lists its own pictures for its card to cycle through, the cover first', () => {
  const db = initDatabase(':memory:');
  const items = [1, 2, 3].map((seed) => add(db, { seed }));
  const lone = add(db, { seed: 9, prompt: 'a blue owl' });
  setGenerationPinned(db, items[0].id, true); // the pinned item becomes the cover

  const list = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { grouped: true });
  const fox = list.find((r) => r.prompt === 'a red fox')!;
  const owl = list.find((r) => r.id === lone.id)!;
  assert.equal(fox.id, items[0].id);
  assert.equal(fox.groupPreviewPaths?.length, 3);
  assert.equal(fox.groupPreviewPaths?.[0], items[0].imagePath);
  assert.deepEqual(new Set(fox.groupPreviewPaths), new Set(items.map((i) => i.imagePath)));
  assert.equal(owl.groupPreviewPaths, undefined, 'a lone item has nothing to cycle');
});

test('a stack previews no more than a dozen pictures', () => {
  const db = initDatabase(':memory:');
  for (let seed = 1; seed <= 20; seed++) add(db, { seed });
  const [stack] = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { grouped: true });
  assert.equal(stack.groupCount, 20);
  assert.equal(stack.groupPreviewPaths?.length, 12);
});
