import { test } from 'node:test';
import assert from 'node:assert/strict';
import { defaultSlotData, newPromptSlot, sanitizeSlots } from './promptSlots';
import { DEFAULT_DENOISE } from './imageToImage';

test('a new tab has no start picture and the default strength', () => {
  const data = defaultSlotData('image');
  assert.equal(data.imageSourcePath, null);
  assert.equal(data.denoise, DEFAULT_DENOISE);
});

test('tabs saved with a start picture and strength are kept', () => {
  const slot = newPromptSlot('image');
  slot.data.imageSourcePath = '/pics/fox.png';
  slot.data.denoise = 0.35;
  const [kept] = sanitizeSlots([slot]);
  assert.equal(kept.data.imageSourcePath, '/pics/fox.png');
  assert.equal(kept.data.denoise, 0.35);
});

test('tabs saved before image to image existed still load, and bad values are refused rather than trusted', () => {
  const old = newPromptSlot('image') as unknown as { id: string; name: null; data: Record<string, unknown> };
  delete old.data.imageSourcePath;
  delete old.data.denoise;
  assert.equal(sanitizeSlots([old]).length, 1);
  assert.equal(sanitizeSlots([old])[0].id, old.id);
  const bad = newPromptSlot('image') as unknown as { id: string; name: null; data: Record<string, unknown> };
  bad.data.denoise = 'a lot';
  // an invalid tab is dropped, and a list with nothing valid in it becomes one fresh tab
  assert.notEqual(sanitizeSlots([bad])[0].id, bad.id);
});
