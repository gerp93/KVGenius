import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as zlib from 'zlib';
import { solidPng } from './solidPng';
import { readPngTextChunks } from './imageMetadata';

test('a solid picture is a valid PNG of the asked size and colour', () => {
  const png = solidPng(4, 3, [10, 20, 30]);
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.equal(png.readUInt32BE(16), 4);
  assert.equal(png.readUInt32BE(20), 3);
  // IDAT sits after IHDR (8 signature + 25 for the IHDR chunk): length, 'IDAT', data
  const idatLength = png.readUInt32BE(33);
  assert.equal(png.toString('latin1', 37, 41), 'IDAT');
  const raw = zlib.inflateSync(png.subarray(41, 41 + idatLength));
  assert.equal(raw.length, 3 * (1 + 4 * 3));
  assert.deepEqual([...raw.subarray(0, 4)], [0, 10, 20, 30]);
  assert.equal(png.toString('latin1', png.length - 8, png.length - 4), 'IEND');
  assert.deepEqual(readPngTextChunks(png), {});
});
