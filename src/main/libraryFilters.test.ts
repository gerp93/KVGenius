import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { countGenerations, initDatabase, insertGeneration, listGenerationRefs, listGenerations, listImageExtensions } from './db';
import { OutputDirs, applyFavorite } from './favorites';
import { GIF_FAMILY } from '../shared/gif';

const VIDEO_FAMILIES = ['wan22-i2v'];
const params = (seed: number) => ({ prompt: `p${seed}`, width: 64, height: 64, seed, steps: 4, cfg: 1 });

function seeded(): DatabaseSync {
  const db = initDatabase(':memory:');
  insertGeneration(db, params(1), 'z-image-turbo', '/out/images/a.png');
  insertGeneration(db, params(2), 'z-image-turbo', '/out/images/b.PNG');
  insertGeneration(db, params(3), 'z-image-turbo', '/out/images/c.webp');
  insertGeneration(db, params(4), GIF_FAMILY, '/out/gifs/d.gif');
  insertGeneration(db, params(5), 'wan22-i2v', '/out/videos/e.mp4');
  return db;
}

test('image listings can be narrowed to one extension, whatever its case', () => {
  const db = seeded();
  const png = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, 'png');
  assert.deepEqual(png.map((r) => r.seed).sort(), [1, 2]);
  const gif = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, 'GIF');
  assert.deepEqual(gif.map((r) => r.seed), [4]);
  assert.equal(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null).length, 4);
  assert.deepEqual(listGenerationRefs(db, VIDEO_FAMILIES, 'image', false, false, 'webp').map((r) => r.imagePath), ['/out/images/c.webp']);
});

test('the extension narrows the image count only, and junk is ignored', () => {
  const db = seeded();
  assert.deepEqual(countGenerations(db, VIDEO_FAMILIES, false, false, 'gif'), { image: 1, video: 1 });
  assert.deepEqual(countGenerations(db, VIDEO_FAMILIES, false, false, null), { image: 4, video: 1 });
  // Not a plain extension: no filtering (and no LIKE wildcards reaching SQL).
  assert.equal(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, '%').length, 4);
});

test('listImageExtensions lists the image types present, most common first, never video types', () => {
  assert.deepEqual(listImageExtensions(seeded(), VIDEO_FAMILIES), ['png', 'gif', 'webp']);
});

test('a favorited GIF lives under gifs/favorites and returns to gifs/', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-gif-'));
  try {
    const dirs: OutputDirs = {
      images: path.join(root, 'images'),
      videos: path.join(root, 'videos'),
      gifs: path.join(root, 'gifs'),
      legacy: path.join(root, 'legacy'),
    };
    fs.mkdirSync(dirs.gifs, { recursive: true });
    const file = path.join(dirs.gifs, 'x.gif');
    fs.writeFileSync(file, 'gif');
    const db = initDatabase(':memory:');
    const id = insertGeneration(db, params(9), GIF_FAMILY, file).id;

    const fav = applyFavorite(db, id, true, dirs, VIDEO_FAMILIES);
    assert.equal(fav.imagePath, path.join(dirs.gifs, 'favorites', 'x.gif'));
    assert.ok(fs.existsSync(fav.imagePath));

    const back = applyFavorite(db, id, false, dirs, VIDEO_FAMILIES);
    assert.equal(back.imagePath, file);
    assert.ok(fs.existsSync(file));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
