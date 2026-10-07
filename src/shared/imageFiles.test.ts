import { test } from 'node:test';
import assert from 'node:assert/strict';
import { IMAGE_EXTENSIONS, isImageFileName } from './imageFiles';

test('isImageFileName accepts the picture formats in any letter case', () => {
  for (const ext of IMAGE_EXTENSIONS) {
    assert.ok(isImageFileName(`cat.${ext}`));
    assert.ok(isImageFileName(`CAT.${ext.toUpperCase()}`));
  }
  assert.ok(isImageFileName(String.raw`C:\pics\my.photo.final.JPG`));
});

test('isImageFileName rejects everything else', () => {
  assert.ok(!isImageFileName('clip.mp4'));
  assert.ok(!isImageFileName('notes.txt'));
  assert.ok(!isImageFileName('png'));
  assert.ok(!isImageFileName('cat.png.exe'));
  assert.ok(!isImageFileName(''));
});
