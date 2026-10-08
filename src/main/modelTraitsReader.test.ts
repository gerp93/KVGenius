import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { readFileTraits, readFolderTraits } from './modelTraitsReader';

/** A real .safetensors layout (8-byte length, JSON header, data) holding tensors of the given shapes with no data in them. */
function writeSafetensors(file: string, shapes: Record<string, number[]>): void {
  const header: Record<string, unknown> = {};
  for (const [name, shape] of Object.entries(shapes)) header[name] = { dtype: 'F16', shape, data_offsets: [0, 0] };
  const json = Buffer.from(JSON.stringify(header), 'utf8');
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(json.length));
  fs.writeFileSync(file, Buffer.concat([length, json]));
}

test('a file is read by its header only, a folder name decides where, and nothing outside the folder is read', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-traits-'));
  try {
    fs.mkdirSync(path.join(root, 'models', 'vae', 'sub'), { recursive: true });
    writeSafetensors(path.join(root, 'models', 'vae', 'ae.safetensors'), { 'decoder.conv_in.weight': [512, 16, 3, 3] });
    writeSafetensors(path.join(root, 'models', 'vae', 'sub', 'old.safetensors'), { 'decoder.conv_in.weight': [512, 4, 3, 3] });
    fs.writeFileSync(path.join(root, 'models', 'vae', 'notes.txt'), 'x');
    writeSafetensors(path.join(root, 'secret.safetensors'), { 'decoder.conv_in.weight': [512, 16, 3, 3] });

    const found = await readFolderTraits(path.join(root, 'models'), 'vae', ['ae.safetensors', 'sub/old.safetensors', 'notes.txt', 'missing.safetensors', '../../secret.safetensors']);
    assert.equal(found['ae.safetensors']?.latentChannels, 16);
    assert.equal(found['sub/old.safetensors']?.latentChannels, 4);
    assert.equal(found['notes.txt'], null, 'not a safetensors file');
    assert.equal(found['missing.safetensors'], null);
    assert.equal(found['../../secret.safetensors'], null, 'a name that leaves the folder is never read');
    assert.deepEqual(await readFolderTraits(path.join(root, 'models'), '../..', ['x']), {}, 'only known folders');
    assert.equal((await readFileTraits(path.join(root, 'models', 'vae', 'ae.safetensors')))?.arch, 'latent-vae');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
