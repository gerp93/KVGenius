import { test } from 'node:test';
import assert from 'node:assert/strict';
import { copyOfSlot, defaultSlotData, newPromptSlot, sanitizeSlots } from './promptSlots';
import { DEFAULT_DENOISE } from './imageToImage';

test('a new tab has no start picture and the default strength', () => {
  const data = defaultSlotData('image');
  assert.equal(data.imageSourcePath, null);
  assert.equal(data.imageFromPicture, false);
  assert.equal(data.denoise, DEFAULT_DENOISE);
});

test('tabs saved with a start picture and strength are kept', () => {
  const slot = newPromptSlot('image');
  slot.data.imageSourcePath = '/pics/fox.png';
  slot.data.imageFromPicture = true;
  slot.data.denoise = 0.35;
  const [kept] = sanitizeSlots([slot]);
  assert.equal(kept.data.imageSourcePath, '/pics/fox.png');
  assert.equal(kept.data.imageFromPicture, true);
  assert.equal(kept.data.denoise, 0.35);
});

test('tabs saved before image to image existed still load, and bad values are refused rather than trusted', () => {
  const old = newPromptSlot('image') as unknown as { id: string; name: null; data: Record<string, unknown> };
  delete old.data.imageSourcePath;
  delete old.data.imageFromPicture;
  delete old.data.denoise;
  assert.equal(sanitizeSlots([old]).length, 1);
  assert.equal(sanitizeSlots([old])[0].id, old.id);
  const bad = newPromptSlot('image') as unknown as { id: string; name: null; data: Record<string, unknown> };
  bad.data.denoise = 'a lot';
  // an invalid tab is dropped, and a list with nothing valid in it becomes one fresh tab
  assert.notEqual(sanitizeSlots([bad])[0].id, bad.id);
});

test('a tab remembers its elements; tabs saved before elements existed have none, and bad ids are refused', () => {
  assert.deepEqual(defaultSlotData('image').elementIds, []);
  const slot = newPromptSlot('image');
  slot.data.elementIds = [3, 7];
  assert.deepEqual(sanitizeSlots([slot])[0].data.elementIds, [3, 7]);
  const old = newPromptSlot('image') as unknown as { id: string; name: null; data: Record<string, unknown> };
  delete old.data.elementIds;
  assert.equal(sanitizeSlots([old])[0].id, old.id);
  const bad = newPromptSlot('image') as unknown as { id: string; name: null; data: Record<string, unknown> };
  bad.data.elementIds = ['x'];
  assert.notEqual(sanitizeSlots([bad])[0].id, bad.id);
});

test('a copy of a tab has its own id and a deep copy of the form, and no last-run record', () => {
  const source = newPromptSlot('video');
  source.data.prompt = 'a fox';
  source.data.elementIds = [1, 2];
  source.data.lastRunSignature = 'sig';
  const copy = copyOfSlot(source);
  assert.notEqual(copy.id, source.id);
  assert.equal(copy.data.mode, 'video');
  assert.equal(copy.data.prompt, 'a fox');
  assert.deepEqual(copy.data.elementIds, [1, 2]);
  assert.equal(copy.data.lastRunSignature, null);
  assert.equal(source.data.lastRunSignature, 'sig');
  copy.data.elementIds.push(3);
  assert.deepEqual(source.data.elementIds, [1, 2]);
  assert.equal(copy.name, null);
});

test("a named tab's copy is named as a copy, within the name limit", () => {
  const source = newPromptSlot();
  source.name = 'Portraits';
  assert.equal(copyOfSlot(source).name, 'Portraits copy');
  source.name = 'x'.repeat(40);
  assert.equal(copyOfSlot(source).name?.length, 40);
});
