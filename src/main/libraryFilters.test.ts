import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { countGenerations, initDatabase, insertGeneration, listGenerationRefs, listGenerations, listImageExtensions } from './db';
import { OutputDirs, applyFavorite } from './favorites';
import { GIF_FAMILY } from '../shared/gif';
import { OriginKind } from '../shared/origin';

const VIDEO_FAMILIES = ['wan22-i2v'];
const params = (seed: number) => ({ prompt: `p${seed}`, width: 64, height: 64, seed, steps: 4, cfg: 1 });

function seeded(): DatabaseSync {
  const db = initDatabase(':memory:');
  insertGeneration(db, params(1), 'z-image', '/out/images/a.png');
  insertGeneration(db, params(2), 'z-image', '/out/images/b.PNG');
  insertGeneration(db, params(3), 'z-image', '/out/images/c.webp');
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

test('a listing can be narrowed to the items made one of the chosen ways', () => {
  const db = initDatabase(':memory:');
  insertGeneration(db, params(1), 'z-image', '/out/images/a.png');
  insertGeneration(db, params(2), 'upscale-image', '/out/images/b.png');
  insertGeneration(db, params(3), GIF_FAMILY, '/out/gifs/c.gif');
  insertGeneration(db, params(4), 'wan22-i2v', '/out/videos/d.mp4');
  insertGeneration(db, params(5), 'upscale-video', '/out/videos/e.mp4');
  const seeds = (origins: OriginKind[], kind: 'image' | 'video') =>
    listGenerations(db, ['wan22-i2v', 'upscale-video'], kind, 50, null, false, false, null, { origins })
      .map((r) => r.seed)
      .sort();
  assert.deepEqual(seeds(['upscale'], 'image'), [2]);
  assert.deepEqual(seeds(['upscale'], 'video'), [5]);
  assert.deepEqual(seeds(['text-to-image', 'gif'], 'image'), [1, 3]);
  assert.deepEqual(seeds(['image-to-video'], 'image'), []);
  // Nothing chosen narrows nothing.
  assert.deepEqual(seeds([], 'image'), [1, 2, 3]);
});

test('the origin filter applies to counts, refs and stacks too, and ignores junk', () => {
  const db = initDatabase(':memory:');
  insertGeneration(db, params(1), 'z-image', '/out/images/a.png');
  insertGeneration(db, params(2), 'upscale-image', '/out/images/b.png');
  const families = ['wan22-i2v', 'upscale-video'];
  assert.deepEqual(countGenerations(db, families, false, false, null, { origins: ['upscale'] }), { image: 1, video: 0 });
  assert.deepEqual(listGenerationRefs(db, families, 'image', false, false, null, { origins: ['text-to-image'] }).length, 1);
  const stacks = listGenerations(db, families, 'image', 50, null, false, false, null, { grouped: true, origins: ['upscale'] });
  assert.deepEqual(stacks.map((r) => r.seed), [2]);
  // Not real origins: no filtering (and nothing reaches SQL).
  const junk = ['x" OR 1=1 --'] as unknown as OriginKind[];
  assert.equal(listGenerations(db, families, 'image', 50, null, false, false, null, { origins: junk }).length, 2);
});

test('counts can carry a separate origin filter for each tab', () => {
  const db = initDatabase(':memory:');
  insertGeneration(db, params(1), 'z-image', '/out/images/a.png');
  insertGeneration(db, params(2), 'upscale-image', '/out/images/b.png');
  insertGeneration(db, params(3), 'wan22-i2v', '/out/videos/c.mp4');
  insertGeneration(db, params(4), 'upscale-video', '/out/videos/d.mp4');
  const families = ['wan22-i2v', 'upscale-video'];
  // Images narrowed to text-to-image, videos left alone.
  assert.deepEqual(countGenerations(db, families, false, false, null, { originsByKind: { image: ['text-to-image'], video: [] } }), { image: 1, video: 2 });
  // Each tab with its own filter.
  assert.deepEqual(
    countGenerations(db, families, false, false, null, { originsByKind: { image: ['upscale'], video: ['image-to-video'] } }),
    { image: 1, video: 1 }
  );
});

test('a search keeps the items whose prompt has every word, ignoring case, in listings, stacks, refs and counts', () => {
  const db = initDatabase(':memory:');
  const add = (prompt: string, family = 'z-image', file = `/out/images/${prompt.length}${Math.random()}.png`) =>
    insertGeneration(db, { prompt, width: 64, height: 64, seed: 1, steps: 4, cfg: 1 }, family, file);
  add('A red fox in the Snow');
  add('A red fox in the Snow');
  add('a blue fox');
  add('100% cotton shirt');
  add('a_b under_score');
  add('a red fox video', 'wan22-i2v', '/out/videos/v.mp4');

  const found = (search: string) => listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { search }).map((r) => r.prompt);
  assert.equal(found('').length, 5, 'blank search narrows nothing');
  assert.equal(found('   ').length, 5);
  assert.deepEqual(found('SNOW'), ['A red fox in the Snow', 'A red fox in the Snow']);
  assert.deepEqual(found('fox snow'), ['A red fox in the Snow', 'A red fox in the Snow'], 'every word, any order');
  assert.deepEqual(found('snow fox'), ['A red fox in the Snow', 'A red fox in the Snow']);
  assert.deepEqual(found('blue red'), []);
  // LIKE wildcards in what is typed are plain characters.
  assert.deepEqual(found('100%'), ['100% cotton shirt']);
  assert.deepEqual(found('%'), ['100% cotton shirt']);
  assert.deepEqual(found('a_b'), ['a_b under_score']);
  assert.deepEqual(found('_'), ['a_b under_score']);

  const stacks = listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, false, null, { search: 'fox', grouped: true });
  assert.deepEqual(stacks.map((r) => [r.prompt, r.groupCount]), [['a blue fox', 1], ['A red fox in the Snow', 2]]);
  assert.equal(listGenerationRefs(db, VIDEO_FAMILIES, 'image', false, false, null, { search: 'snow' }).length, 2);
  assert.deepEqual(countGenerations(db, VIDEO_FAMILIES, false, false, null, { search: 'fox' }), { image: 3, video: 1 });
  assert.deepEqual(countGenerations(db, VIDEO_FAMILIES, false, false, null, { search: 'fox', grouped: true }), { image: 2, video: 1 });
});
