import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { saveMaskPng } from './maskStore';

// A 1x1 black PNG.
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';

test('a painted mask is stored once, by its contents', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-mask-'));
  try {
    const a = saveMaskPng(`data:image/png;base64,${PNG}`, dir);
    const b = saveMaskPng(`data:image/png;base64,${PNG}`, dir);
    assert.equal(a, b);
    assert.ok(a.startsWith(dir) && a.endsWith('.png'));
    assert.equal(fs.readdirSync(dir).length, 1);
    assert.ok(fs.readFileSync(a).length > 8);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('anything that is not a PNG data URL is refused and nothing is written', () => {
  const dir = path.join(os.tmpdir(), `kvg-mask-none-${process.pid}`);
  assert.throws(() => saveMaskPng(42, dir), /not an image/);
  assert.throws(() => saveMaskPng('data:image/jpeg;base64,AAAA', dir), /PNG/);
  assert.throws(() => saveMaskPng('data:image/png;base64,../../etc/passwd', dir), /PNG/);
  // valid base64 and prefix, but not a PNG inside
  assert.throws(() => saveMaskPng(`data:image/png;base64,${Buffer.from('hello world').toString('base64')}`, dir), /PNG/);
  assert.equal(fs.existsSync(dir), false);
});
