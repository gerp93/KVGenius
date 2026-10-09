import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { thumbnailFor, thumbnailWidth } from './thumbnails';
import { handleMediaRequest } from './mediaProtocol';

function setup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kv-thumbs-'));
  const picture = path.join(dir, 'out', 'a.png');
  fs.mkdirSync(path.dirname(picture));
  fs.writeFileSync(picture, Buffer.alloc(5000, 1));
  return { dir, picture, cache: path.join(dir, 'cache') };
}

test('only whole widths in range are accepted', () => {
  assert.equal(thumbnailWidth('800'), 800);
  assert.equal(thumbnailWidth('63'), null);
  assert.equal(thumbnailWidth('2049'), null);
  assert.equal(thumbnailWidth('80x'), null);
  assert.equal(thumbnailWidth(null), null);
});

test('a thumbnail is made once, kept, and made again when the picture changes', async () => {
  const { picture, cache } = setup();
  let made = 0;
  const resize = async () => {
    made++;
    return Buffer.from(`thumb${made}`);
  };
  const first = await thumbnailFor(picture, 800, cache, resize);
  assert.equal(first?.data.toString(), 'thumb1');
  assert.equal(first?.mime, 'image/jpeg');
  assert.equal((await thumbnailFor(picture, 800, cache, resize))?.data.toString(), 'thumb1', 'served from the cache');
  assert.equal(made, 1);
  assert.equal((await thumbnailFor(picture, 400, cache, resize))?.data.toString(), 'thumb2', 'another width is another copy');

  fs.writeFileSync(picture, Buffer.alloc(6000, 2));
  assert.equal((await thumbnailFor(picture, 800, cache, resize))?.data.toString(), 'thumb3', 'an edited picture gets a new copy');
  assert.deepEqual(fs.readdirSync(cache).filter((f) => f.endsWith('.part')), [], 'no half-written files are left');
});

test('what cannot or need not be shrunk is left to the original', async () => {
  const { dir, picture, cache } = setup();
  assert.equal(await thumbnailFor(picture, 800, cache, async () => null), null, 'already small');
  assert.equal(await thumbnailFor(picture, 800, cache, async () => { throw new Error('unreadable'); }), null);
  assert.equal(await thumbnailFor(path.join(dir, 'missing.png'), 800, cache, async () => Buffer.from('x')), null);
  const gif = path.join(dir, 'out', 'a.gif');
  fs.writeFileSync(gif, 'GIF89a');
  assert.equal(await thumbnailFor(gif, 800, cache, async () => Buffer.from('x')), null, 'an animated GIF is served as it is');
});

test('a ?w= request on the media protocol gets the small copy; without one, or with no copy to give, the file itself', async () => {
  const { dir, picture } = setup();
  const url = `kvimage://${encodeURIComponent(picture)}`;
  const allowed = [path.join(dir, 'out')];
  const thumb = async () => ({ data: Buffer.from('tiny'), mime: 'image/jpeg' });

  const small = await handleMediaRequest(new Request(`${url}?w=800`), allowed, new Set(), thumb);
  assert.equal(small.headers.get('content-type'), 'image/jpeg');
  assert.equal(await small.text(), 'tiny');

  const full = await handleMediaRequest(new Request(url), allowed, new Set(), thumb);
  assert.equal((await full.arrayBuffer()).byteLength, 5000);

  const none = await handleMediaRequest(new Request(`${url}?w=800`), allowed, new Set(), async () => null);
  assert.equal((await none.arrayBuffer()).byteLength, 5000, 'no copy: the original is served');

  const badWidth = await handleMediaRequest(new Request(`${url}?w=5`), allowed, new Set(), thumb);
  assert.equal((await badWidth.arrayBuffer()).byteLength, 5000);

  const outside = await handleMediaRequest(new Request(`kvimage://${encodeURIComponent(path.join(dir, 'x.png'))}?w=800`), allowed, new Set(), thumb);
  assert.equal(outside.status, 403, 'the folder rule still applies');
});
