import * as fs from 'fs';
import { faststartFile } from './mp4Faststart';

export interface VideoDiagnosis {
  /** Human-readable findings about the file, one per line. */
  lines: string[];
  /** True when the file was rewritten, so the caller should reload it. */
  repaired: boolean;
}

const CODEC_TAGS: [string, string][] = [
  ['avc1', 'H.264'],
  ['avc3', 'H.264'],
  ['hvc1', 'HEVC (H.265)'],
  ['hev1', 'HEVC (H.265)'],
  ['vp09', 'VP9'],
  ['av01', 'AV1'],
];

/** Top-level box layout, read from headers only (never the media itself). */
async function readTopLevelBoxes(handle: fs.promises.FileHandle, size: number) {
  const boxes: { type: string; offset: number; size: number }[] = [];
  const header = Buffer.alloc(16);
  let offset = 0;
  while (offset + 8 <= size && boxes.length < 24) {
    await handle.read(header, 0, 16, offset);
    let boxSize = header.readUInt32BE(0);
    const type = header.toString('latin1', 4, 8);
    if (boxSize === 1) boxSize = Number(header.readBigUInt64BE(8));
    else if (boxSize === 0) boxSize = size - offset;
    boxes.push({ type, offset, size: boxSize });
    if (boxSize < 8) break;
    offset += boxSize;
  }
  return { boxes, endedAt: offset };
}

/**
 * Explains why the in-app player might be rejecting a video, and tries the one fix we have: if the
 * MP4's index is at the end of the file, rewrite it with the index up front. Reads box headers and
 * the codec description only.
 */
export async function diagnoseVideo(filePath: string): Promise<VideoDiagnosis> {
  const lines: string[] = [];
  let repaired = false;
  let indexAtEnd = false;

  let handle: fs.promises.FileHandle;
  try {
    handle = await fs.promises.open(filePath, 'r');
  } catch (err) {
    return { lines: [`File could not be opened: ${err instanceof Error ? err.message : String(err)}`], repaired };
  }

  try {
    const { size } = await handle.stat();
    lines.push(`File size: ${size} bytes`);

    const { boxes, endedAt } = await readTopLevelBoxes(handle, size);
    lines.push(`Layout: ${boxes.map((b) => `${b.type}(${b.size})`).join(' ')}`);
    if (endedAt !== size) lines.push(`Layout does not end at the end of the file (parsed to ${endedAt} of ${size}).`);

    const moov = boxes.find((b) => b.type === 'moov');
    const mdat = boxes.find((b) => b.type === 'mdat');
    if (boxes.some((b) => b.type === 'moof')) lines.push('Fragmented MP4 (moof boxes present).');
    if (!moov) lines.push('No moov box: this is not a complete MP4.');
    else if (mdat) {
      indexAtEnd = moov.offset > mdat.offset;
      lines.push(indexAtEnd ? 'Index (moov) is after the media data.' : 'Index (moov) is before the media data: OK.');
    }

    if (moov && moov.size <= 8 * 1024 * 1024) {
      const buf = Buffer.alloc(moov.size);
      await handle.read(buf, 0, moov.size, moov.offset);
      const text = buf.toString('latin1');
      const codec = CODEC_TAGS.find(([tag]) => text.includes(tag));
      lines.push(`Codec: ${codec ? codec[1] : 'unrecognised'}`);
      const avcC = buf.indexOf('avcC', 0, 'latin1');
      if (avcC >= 0 && avcC + 9 < buf.length) {
        lines.push(`H.264 profile ${buf[avcC + 5]}, level ${buf[avcC + 7]} (66 = Baseline, 77 = Main, 100 = High, 110+ = 10-bit/4:2:2/4:4:4)`);
      }
    }
  } finally {
    await handle.close();
  }

  try {
    repaired = await faststartFile(filePath);
    if (repaired) lines.push('Repair: index moved to the front of the file.');
    else if (indexAtEnd) lines.push('Repair: could not rewrite this file (layout not recognised).');
    else lines.push('Repair: nothing to change.');
  } catch (err) {
    lines.push(`Repair failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  return { lines, repaired };
}
