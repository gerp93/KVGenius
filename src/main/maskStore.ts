import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

const MAX_MASK_BYTES = 30 * 1024 * 1024;
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/**
 * Stores a mask painted in the app (a PNG data URL from the editor's canvas) in `dir`, named by a hash of its contents
 * so the same mask is stored once, and returns its path. Only a real PNG of a sensible size is accepted - the renderer
 * hands over arbitrary text, and this is the one place it becomes a file.
 */
export function saveMaskPng(dataUrl: unknown, dir: string): string {
  if (typeof dataUrl !== 'string') throw new Error('The mask was not an image.');
  const match = /^data:image\/png;base64,([A-Za-z0-9+/]+={0,2})$/.exec(dataUrl);
  if (!match) throw new Error('The mask was not a PNG image.');
  const bytes = Buffer.from(match[1], 'base64');
  if (bytes.length < PNG_SIGNATURE.length || !bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    throw new Error('The mask was not a PNG image.');
  }
  if (bytes.length > MAX_MASK_BYTES) throw new Error('The mask is too large.');
  const hash = crypto.createHash('sha1').update(bytes).digest('hex').slice(0, 20);
  fs.mkdirSync(dir, { recursive: true });
  const target = path.join(dir, `${hash}.png`);
  if (!fs.existsSync(target)) fs.writeFileSync(target, bytes);
  return target;
}
