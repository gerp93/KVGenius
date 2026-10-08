import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { getGenerationById, initDatabase, insertGeneration, listKeptSources } from './db';
import { keepSourceImage, releaseSourceImage } from './sourceImages';

const temp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-src-'));
const params = (seed = 1) => ({ prompt: 'p', width: 64, height: 64, seed, steps: 4, cfg: 1, length: 81 });

test('a source image is copied into the sources folder, named by its contents', () => {
  const dir = temp();
  const sources = path.join(dir, 'sources');
  const original = path.join(dir, 'Elsewhere.PNG');
  fs.writeFileSync(original, 'picture-bytes');

  const kept = keepSourceImage(original, sources);
  assert.ok(kept);
  assert.equal(path.dirname(kept), sources);
  assert.equal(path.extname(kept), '.png');
  assert.equal(fs.readFileSync(kept, 'utf8'), 'picture-bytes');

  // The copy outlives the original - the whole point.
  fs.rmSync(original);
  assert.equal(fs.readFileSync(kept, 'utf8'), 'picture-bytes');
});

test('the same picture is kept once, however many videos use it', () => {
  const dir = temp();
  const sources = path.join(dir, 'sources');
  const a = path.join(dir, 'a.png');
  const b = path.join(dir, 'moved-and-renamed.png');
  const other = path.join(dir, 'other.png');
  fs.writeFileSync(a, 'same');
  fs.writeFileSync(b, 'same');
  fs.writeFileSync(other, 'different');

  const first = keepSourceImage(a, sources);
  const second = keepSourceImage(b, sources);
  assert.equal(first, second);
  assert.notEqual(keepSourceImage(other, sources), first);
  assert.equal(fs.readdirSync(sources).length, 2);
});

test('a file already in the sources folder is its own copy, and a missing one is not kept', () => {
  const dir = temp();
  const sources = path.join(dir, 'sources');
  const kept = keepSourceImage((() => {
    const file = path.join(dir, 'x.png');
    fs.writeFileSync(file, 'x');
    return file;
  })(), sources);
  assert.ok(kept);
  assert.equal(keepSourceImage(kept, sources), path.resolve(kept));
  assert.equal(fs.readdirSync(sources).length, 1);

  assert.equal(keepSourceImage(path.join(dir, 'gone.png'), sources), null);
});

test('a generation remembers its source image, and older ones have none', () => {
  const db = initDatabase(':memory:');
  const video = insertGeneration(db, params(1), 'wan22-i2v', '/out/v.mp4', null, false, '/out/sources/abc.png');
  const image = insertGeneration(db, params(2), 'z-image', '/out/i.png');
  assert.equal(video.sourceImagePath, '/out/sources/abc.png');
  assert.equal(getGenerationById(db, video.id)?.sourceImagePath, '/out/sources/abc.png');
  assert.equal(getGenerationById(db, image.id)?.sourceImagePath, null);
});

test('a kept source image is deleted with its last video, and nothing outside the sources folder is touched', () => {
  const dir = temp();
  const sources = path.join(dir, 'sources');
  const original = path.join(dir, 'a.png');
  fs.writeFileSync(original, 'bytes');
  const kept = keepSourceImage(original, sources)!;

  const db = initDatabase(':memory:');
  const one = insertGeneration(db, params(1), 'wan22-i2v', '/out/1.mp4', null, false, kept);
  const two = insertGeneration(db, params(2), 'wan22-i2v', '/out/2.mp4', null, false, kept);

  db.prepare('DELETE FROM generations WHERE id = ?').run(one.id);
  assert.equal(releaseSourceImage(db, kept, sources), false, 'another video still uses it');
  assert.ok(fs.existsSync(kept));

  db.prepare('DELETE FROM generations WHERE id = ?').run(two.id);
  assert.equal(releaseSourceImage(db, kept, sources), true);
  assert.equal(fs.existsSync(kept), false);

  // A path outside the sources folder (an old record, a hand-edited row) is never deleted.
  assert.equal(releaseSourceImage(db, original, sources), false);
  assert.ok(fs.existsSync(original));
  assert.equal(releaseSourceImage(db, null, sources), false);
});

test('listKeptSources counts what each kept source was used for, newest use first', () => {
  const db = initDatabase(':memory:');
  insertGeneration(db, params(1), 'wan22-i2v', '/out/1.mp4', null, false, '/s/a.png');
  insertGeneration(db, params(2), 'upscale-image', '/out/2.png', null, false, '/s/a.png');
  insertGeneration(db, params(3), 'wan22-i2v', '/out/3.mp4', null, false, '/s/b.png');
  insertGeneration(db, params(4), 'z-image', '/out/4.png', null, false, null);

  const sources = listKeptSources(db);
  assert.deepEqual(
    sources.map((s) => [s.path, s.uses]),
    [
      ['/s/b.png', 1],
      ['/s/a.png', 2],
    ]
  );
  assert.deepEqual(sources[1].families.sort(), ['upscale-image', 'wan22-i2v']);
});

test('an inpainting result remembers its mask, which is kept as long as any result uses it as a mask or a source', () => {
  const dir = temp();
  const sources = path.join(dir, 'sources');
  fs.mkdirSync(sources, { recursive: true });
  const picture = path.join(sources, 'pic.png');
  const mask = path.join(sources, 'mask.png');
  fs.writeFileSync(picture, 'p');
  fs.writeFileSync(mask, 'm');

  const db = initDatabase(':memory:');
  const result = insertGeneration(db, { ...params(1), denoise: 0.8 }, 'z-image-inpaint', '/out/1.png', null, false, picture, mask);
  assert.equal(result.maskImagePath, mask);
  assert.equal(getGenerationById(db, result.id)?.maskImagePath, mask);
  assert.equal(getGenerationById(db, insertGeneration(db, params(2), 'z-image', '/out/2.png').id)?.maskImagePath, null);

  assert.equal(releaseSourceImage(db, mask, sources), false, 'still the mask of a result');
  db.prepare('DELETE FROM generations WHERE id = ?').run(result.id);
  assert.equal(releaseSourceImage(db, mask, sources), true);
  assert.equal(releaseSourceImage(db, picture, sources), true);
});
