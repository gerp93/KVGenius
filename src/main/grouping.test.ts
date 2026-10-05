import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import {
  countGenerations,
  initDatabase,
  insertGeneration,
  listGenerationRefs,
  listGenerations,
  setGenerationFavorite,
  setGenerationHidden,
  setGenerationPinned,
} from './db';
import { moveToTrash } from './trash';

const VIDEO_FAMILIES = ['wan22-i2v'];
let seed = 0;

function add(db: DatabaseSync, prompt: string, family = 'z-image-turbo'): number {
  return insertGeneration(
    db,
    { prompt, width: 8, height: 8, seed: ++seed, steps: 1, cfg: 1, ...(family === 'wan22-i2v' ? { length: 81 } : {}) },
    family,
    `/out/${seed}.${family === 'wan22-i2v' ? 'mp4' : 'png'}`
  ).id;
}

const stacks = (db: DatabaseSync, limit = 50, before: number | null = null, fav = false, hidden = false) =>
  listGenerations(db, VIDEO_FAMILIES, 'image', limit, before, fav, hidden, null, { grouped: true });

test('items with exactly the same prompt collapse into one stack, with how many it holds', () => {
  const db = initDatabase(':memory:');
  const a1 = add(db, 'a red fox');
  const a2 = add(db, 'a red fox');
  const b = add(db, 'a blue owl');
  const a3 = add(db, 'a red fox');

  const list = stacks(db);
  assert.equal(list.length, 2);
  const fox = list.find((r) => r.prompt === 'a red fox')!;
  const owl = list.find((r) => r.prompt === 'a blue owl')!;
  assert.equal(fox.groupCount, 3);
  assert.equal(owl.groupCount, 1);
  assert.equal(fox.id, a3, 'with nothing marked, the newest is the cover');
  assert.equal(owl.id, b);
  assert.deepEqual([a1, a2].some((id) => list.some((r) => r.id === id)), false);
});

test('"the same" means exactly the same - not a different case or an extra space', () => {
  const db = initDatabase(':memory:');
  add(db, 'a red fox');
  add(db, 'A red fox');
  add(db, 'a red fox ');
  add(db, 'a  red fox');
  assert.equal(stacks(db).length, 4);
  assert.equal(countGenerations(db, VIDEO_FAMILIES, false, true, null, { grouped: true }).image, 4);
});

test('the cover is the pinned item, else a favorite, else the newest', () => {
  const db = initDatabase(':memory:');
  const oldest = add(db, 'p');
  const middle = add(db, 'p');
  const newest = add(db, 'p');
  assert.equal(stacks(db)[0].id, newest);

  setGenerationFavorite(db, middle, true);
  assert.equal(stacks(db)[0].id, middle, 'a favorite beats the newest');

  setGenerationPinned(db, oldest, true);
  assert.equal(stacks(db)[0].id, oldest, 'a pinned item beats a favorite');
  assert.equal(stacks(db)[0].groupCount, 3);
});

test('stacks are ordered by their newest item, not by their cover', () => {
  const db = initDatabase(':memory:');
  const oldPinned = add(db, 'first look');
  add(db, 'second look');
  const newestOfFirst = add(db, 'first look');
  setGenerationPinned(db, oldPinned, true);

  const list = stacks(db);
  assert.deepEqual(list.map((r) => r.prompt), ['first look', 'second look'], 'the stack with the latest activity leads');
  assert.equal(list[0].id, oldPinned, 'although its cover is the older, pinned item');
  assert.equal(list[0].groupNewestId, newestOfFirst);
});

test('stacks page by their newest item with nothing repeated or skipped', () => {
  const db = initDatabase(':memory:');
  for (let i = 0; i < 7; i++) {
    add(db, `prompt ${i}`);
    add(db, `prompt ${i}`);
  }
  const seen: string[] = [];
  let before: number | null = null;
  for (;;) {
    const page = stacks(db, 3, before);
    if (page.length === 0) break;
    seen.push(...page.map((r) => r.prompt));
    before = page[page.length - 1].groupNewestId!;
  }
  assert.equal(seen.length, 7);
  assert.equal(new Set(seen).size, 7);
  assert.deepEqual(seen, ['prompt 6', 'prompt 5', 'prompt 4', 'prompt 3', 'prompt 2', 'prompt 1', 'prompt 0']);
});

test('the filters apply to the items first, so a stack only holds what passes them', () => {
  const db = initDatabase(':memory:');
  const kept = add(db, 'shared');
  add(db, 'shared');
  const hiddenOne = add(db, 'shared');
  const onlyHidden = add(db, 'secret');
  setGenerationHidden(db, hiddenOne, true);
  setGenerationHidden(db, onlyHidden, true);
  setGenerationFavorite(db, kept, true);

  const visible = stacks(db);
  assert.deepEqual(visible.map((r) => [r.prompt, r.groupCount]), [['shared', 2]], 'hidden items are left out, and a stack of only hidden ones vanishes');
  assert.equal(stacks(db, 50, null, false, true).find((r) => r.prompt === 'shared')!.groupCount, 3);
  assert.deepEqual(stacks(db, 50, null, true, false).map((r) => [r.prompt, r.groupCount]), [['shared', 1]], 'favorites only');
});

test('images and videos stack separately, so a prompt used for both appears in each tab once', () => {
  const db = initDatabase(':memory:');
  add(db, 'both ways');
  add(db, 'both ways');
  add(db, 'both ways', 'wan22-i2v');
  assert.equal(stacks(db).length, 1);
  assert.equal(stacks(db)[0].groupCount, 2);
  const videos = listGenerations(db, VIDEO_FAMILIES, 'video', 50, null, false, true, null, { grouped: true });
  assert.equal(videos.length, 1);
  assert.equal(videos[0].groupCount, 1);
  assert.deepEqual(countGenerations(db, VIDEO_FAMILIES, false, true, null, { grouped: true }), { image: 1, video: 1 });
});

test('anything in the Trash is not part of a stack', () => {
  const db = initDatabase(':memory:');
  add(db, 'p');
  const second = add(db, 'p');
  const alone = add(db, 'lonely');
  assert.equal(stacks(db).find((r) => r.prompt === 'p')!.groupCount, 2);
  moveToTrash(db, [second, alone], '/nonexistent-trash-dir-for-test-' + Date.now());
  const list = stacks(db);
  assert.deepEqual(list.map((r) => [r.prompt, r.groupCount]), [['p', 1]]);
});

test('without grouping the listing is exactly what it always was', () => {
  const db = initDatabase(':memory:');
  add(db, 'x');
  add(db, 'x');
  const plain = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, true);
  assert.equal(plain.length, 2);
  assert.equal(plain[0].groupCount, undefined);
  assert.equal(countGenerations(db, VIDEO_FAMILIES, false, true).image, 2);
});

test('opening a stack lists just that prompt\'s items, newest first, and refs and counts follow', () => {
  const db = initDatabase(':memory:');
  const a = add(db, 'target');
  add(db, 'other');
  const b = add(db, 'target');
  const hidden = add(db, 'target');
  setGenerationHidden(db, hidden, true);

  const opened = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { prompt: 'target' });
  assert.deepEqual(opened.map((r) => r.id), [b, a]);
  assert.equal(opened[0].groupCount, undefined, 'an opened stack is a plain list');
  assert.deepEqual(listGenerationRefs(db, VIDEO_FAMILIES, 'image', false, false, null, { prompt: 'target' }).map((r) => r.id), [b, a]);
  assert.equal(countGenerations(db, VIDEO_FAMILIES, false, false, null, { prompt: 'target' }).image, 2);
  // Asking to group and to open one prompt opens it.
  assert.equal(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { prompt: 'target', grouped: true }).length, 2);
  // A prompt that is not there lists nothing.
  assert.deepEqual(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { prompt: 'no such prompt' }), []);
});
