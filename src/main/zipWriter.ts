import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';

export interface ZipSource {
  /** File on disk to add. */
  sourcePath: string;
  /** Path inside the archive. */
  name: string;
}

const MAX_32 = 0xffffffff;

function dosDateTime(date: Date): { time: number; date: number } {
  const year = Math.max(1980, date.getFullYear());
  return {
    time: (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2),
    date: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
  };
}

/** Archive names must be unique: `a.png`, `a.png` -> `a.png`, `a-2.png`. */
export function uniqueNames(fileNames: string[]): string[] {
  const used = new Set<string>();
  return fileNames.map((original) => {
    const ext = path.extname(original);
    const stem = original.slice(0, original.length - ext.length);
    let candidate = original;
    for (let n = 2; used.has(candidate.toLowerCase()); n++) candidate = `${stem}-${n}${ext}`;
    used.add(candidate.toLowerCase());
    return candidate;
  });
}

/**
 * Writes a ZIP archive with the files stored uncompressed (images and videos are already
 * compressed, so deflating them just burns time). Files are streamed one at a time, so memory
 * stays flat however big the export is. Plain ZIP, not ZIP64: refuses anything over 4 GB or
 * 65,535 files rather than writing a corrupt archive. Deletes the partial file on any failure.
 */
export async function writeZip(destPath: string, sources: ZipSource[]): Promise<void> {
  if (sources.length > 0xffff) throw new Error('Too many files for one zip (limit 65,535).');

  const out = fs.createWriteStream(destPath);
  const write = (buf: Buffer) =>
    new Promise<void>((resolve, reject) => out.write(buf, (err) => (err ? reject(err) : resolve())));

  try {
    let offset = 0;
    const central: Buffer[] = [];

    for (const source of sources) {
      const stat = await fs.promises.stat(source.sourcePath);
      if (stat.size > MAX_32) throw new Error(`${source.name} is over 4 GB, too big for a zip.`);
      const nameBytes = Buffer.from(source.name, 'utf8');
      const stamp = dosDateTime(stat.mtime);
      // bit 3: sizes/CRC follow the data; bit 11: names are UTF-8.
      const flags = 0x0808;

      const local = Buffer.alloc(30);
      local.writeUInt32LE(0x04034b50, 0);
      local.writeUInt16LE(20, 4);
      local.writeUInt16LE(flags, 6);
      local.writeUInt16LE(0, 8); // stored
      local.writeUInt16LE(stamp.time, 10);
      local.writeUInt16LE(stamp.date, 12);
      local.writeUInt16LE(nameBytes.length, 26);
      const headerOffset = offset;
      await write(local);
      await write(nameBytes);
      offset += local.length + nameBytes.length;

      let crc = 0;
      let size = 0;
      for await (const chunk of fs.createReadStream(source.sourcePath)) {
        const data = chunk as Buffer;
        crc = zlib.crc32(data, crc);
        size += data.length;
        await write(data);
      }
      offset += size;
      if (offset > MAX_32) throw new Error('The export is over 4 GB, too big for a zip.');

      const descriptor = Buffer.alloc(16);
      descriptor.writeUInt32LE(0x08074b50, 0);
      descriptor.writeUInt32LE(crc >>> 0, 4);
      descriptor.writeUInt32LE(size, 8);
      descriptor.writeUInt32LE(size, 12);
      await write(descriptor);
      offset += descriptor.length;

      const entry = Buffer.alloc(46);
      entry.writeUInt32LE(0x02014b50, 0);
      entry.writeUInt16LE(20, 4);
      entry.writeUInt16LE(20, 6);
      entry.writeUInt16LE(flags, 8);
      entry.writeUInt16LE(0, 10);
      entry.writeUInt16LE(stamp.time, 12);
      entry.writeUInt16LE(stamp.date, 14);
      entry.writeUInt32LE(crc >>> 0, 16);
      entry.writeUInt32LE(size, 20);
      entry.writeUInt32LE(size, 24);
      entry.writeUInt16LE(nameBytes.length, 28);
      entry.writeUInt32LE(headerOffset, 42);
      central.push(entry, nameBytes);
    }

    const centralSize = central.reduce((sum, b) => sum + b.length, 0);
    for (const part of central) await write(part);

    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(sources.length, 8);
    end.writeUInt16LE(sources.length, 10);
    end.writeUInt32LE(centralSize, 12);
    end.writeUInt32LE(offset, 16);
    await write(end);

    await new Promise<void>((resolve, reject) => {
      out.once('error', reject);
      out.end(resolve);
    });
  } catch (err) {
    out.destroy();
    await fs.promises.rm(destPath, { force: true });
    throw err;
  }
}
