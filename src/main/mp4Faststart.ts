import * as fs from 'fs';

/**
 * Chromium can't play a large MP4 whose index (the `moov` box) sits after the media data when the
 * file is served through the app's custom `kvimage://` protocol - it needs the tail of the file
 * first and gives up with "format not supported". ComfyUI writes exactly that layout. Moving
 * `moov` in front of `mdat` ("faststart", what `ffmpeg -movflags +faststart` does) fixes it, so
 * the chunk offsets inside `moov` are shifted to match.
 */

interface Box {
  type: string;
  start: number;
  size: number;
}

/** Boxes that only contain other boxes and lead down to the chunk-offset tables. */
const CONTAINERS = new Set(['moov', 'trak', 'mdia', 'minf', 'stbl']);

function readBoxes(buf: Buffer, from: number, to: number): Box[] | null {
  const boxes: Box[] = [];
  let offset = from;
  while (offset + 8 <= to) {
    let size = buf.readUInt32BE(offset);
    const type = buf.toString('latin1', offset + 4, offset + 8);
    if (size === 1) {
      if (offset + 16 > to) return null;
      size = Number(buf.readBigUInt64BE(offset + 8));
    } else if (size === 0) {
      size = to - offset;
    }
    if (size < 8 || offset + size > to) return null;
    boxes.push({ type, start: offset, size });
    offset += size;
  }
  if (offset === to) return boxes;
  // Some writers pad the end of the file with zero bytes; that is not part of any box.
  return buf.subarray(offset, to).every((byte) => byte === 0) ? boxes : null;
}

/** Adds `delta` to every stco/co64 chunk offset under `moov` (in place). Returns false if an
 * offset would overflow its field or the structure isn't what we expect. */
function shiftChunkOffsets(moov: Buffer, from: number, to: number, delta: number, mdatStart: number): boolean {
  const boxes = readBoxes(moov, from, to);
  if (!boxes) return false;
  for (const box of boxes) {
    const contentStart = box.start + 8;
    const end = box.start + box.size;
    if (CONTAINERS.has(box.type)) {
      if (!shiftChunkOffsets(moov, contentStart, end, delta, mdatStart)) return false;
    } else if (box.type === 'stco' || box.type === 'co64') {
      const wide = box.type === 'co64';
      const count = moov.readUInt32BE(contentStart + 4);
      const entrySize = wide ? 8 : 4;
      let pos = contentStart + 8;
      if (pos + count * entrySize > end) return false;
      for (let i = 0; i < count; i++, pos += entrySize) {
        if (wide) {
          const value = moov.readBigUInt64BE(pos);
          if (value >= BigInt(mdatStart)) moov.writeBigUInt64BE(value + BigInt(delta), pos);
        } else {
          const value = moov.readUInt32BE(pos);
          if (value >= mdatStart) {
            if (value + delta > 0xffffffff) return false;
            moov.writeUInt32BE(value + delta, pos);
          }
        }
      }
    }
  }
  return true;
}

/** Returns a copy of the MP4 with `moov` moved ahead of `mdat`, or null when it already is (or
 * the file is fragmented / not something we can safely rewrite). */
export function faststartMp4(input: Buffer): Buffer | null {
  const boxes = readBoxes(input, 0, input.length);
  if (!boxes) return null;
  if (boxes.some((b) => b.type === 'moof')) return null;

  const mdat = boxes.find((b) => b.type === 'mdat');
  const moov = boxes.find((b) => b.type === 'moov');
  if (!mdat || !moov || moov.start < mdat.start) return null;

  const moovBytes = Buffer.from(input.subarray(moov.start, moov.start + moov.size));
  if (!shiftChunkOffsets(moovBytes, 8, moovBytes.length, moov.size, mdat.start)) return null;

  const parts: Buffer[] = [];
  for (const box of boxes) {
    if (box === moov) continue;
    if (box === mdat) parts.push(moovBytes);
    parts.push(input.subarray(box.start, box.start + box.size));
  }
  return Buffer.concat(parts);
}

/** Cheap check (reads only box headers) for an MP4 whose `moov` comes after `mdat`. */
export async function needsFaststart(filePath: string): Promise<boolean> {
  const handle = await fs.promises.open(filePath, 'r');
  try {
    const { size } = await handle.stat();
    const header = Buffer.alloc(16);
    let offset = 0;
    let sawMdat = false;
    while (offset + 8 <= size) {
      await handle.read(header, 0, 16, offset);
      let boxSize = header.readUInt32BE(0);
      const type = header.toString('latin1', 4, 8);
      if (boxSize === 1) boxSize = Number(header.readBigUInt64BE(8));
      else if (boxSize === 0) boxSize = size - offset;
      if (type === 'moof') return false;
      if (type === 'mdat') sawMdat = true;
      if (type === 'moov') return sawMdat;
      if (boxSize < 8) return false;
      offset += boxSize;
    }
    return false;
  } finally {
    await handle.close();
  }
}

/** Rewrites an MP4 in place (via a temp file) so its index is up front. Returns true if the file
 * was changed. Never leaves a half-written file behind. */
export async function faststartFile(filePath: string): Promise<boolean> {
  if (!(await needsFaststart(filePath))) return false;
  const fixed = faststartMp4(await fs.promises.readFile(filePath));
  if (!fixed) return false;
  const tempPath = `${filePath}.faststart.tmp`;
  try {
    await fs.promises.writeFile(tempPath, fixed);
    await fs.promises.rename(tempPath, filePath);
    return true;
  } catch (err) {
    await fs.promises.rm(tempPath, { force: true });
    throw err;
  }
}
