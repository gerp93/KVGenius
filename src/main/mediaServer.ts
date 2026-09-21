import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import { randomBytes } from 'crypto';
import { MIME_BY_EXTENSION, isAllowedMediaPath } from './mediaProtocol';

export interface MediaServer {
  /** e.g. `http://127.0.0.1:53211/<token>` - append `/<encoded file path>`. */
  base: string;
  close: () => void;
}

/**
 * Serves generated videos over plain HTTP on the loopback interface.
 *
 * Videos go through here instead of the `kvimage://` custom protocol because Chromium's media
 * player is built around ordinary HTTP range requests: through the custom protocol a paused video
 * (the metadata load of a thumbnail, or a player waiting for the user to press play) had its
 * response go stale, and playback then died a few frames in with "PIPELINE_ERROR_READ: data
 * source error". Only 127.0.0.1 is bound, every URL carries a random per-launch token, and only
 * files inside the allowed folders (or individually allowed files) are served.
 */
export function startMediaServer(
  getAllowedDirectories: () => string[],
  extraAllowedFiles: ReadonlySet<string>
): Promise<MediaServer> {
  const token = randomBytes(16).toString('hex');

  const server = http.createServer((req, res) => {
    const fail = (status: number) => {
      res.statusCode = status;
      res.end();
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') return fail(405);

    let filePath: string;
    try {
      const segments = new URL(req.url ?? '/', 'http://127.0.0.1').pathname.split('/').filter(Boolean);
      if (segments.length !== 2 || segments[0] !== token) return fail(403);
      filePath = path.resolve(decodeURIComponent(segments[1]));
    } catch {
      return fail(400);
    }
    if (!isAllowedMediaPath(filePath, getAllowedDirectories(), extraAllowedFiles)) return fail(403);

    fs.stat(filePath, (err, stat) => {
      if (err || !stat.isFile()) return fail(404);
      const size = stat.size;

      let start = 0;
      let end = size - 1;
      let status = 200;
      const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
      if (match && (match[1] !== '' || match[2] !== '')) {
        if (match[1] === '') {
          start = Math.max(0, size - Number(match[2]));
        } else {
          start = Number(match[1]);
          if (match[2] !== '') end = Math.min(Number(match[2]), size - 1);
        }
        if (start > end || start >= size) {
          res.setHeader('Content-Range', `bytes */${size}`);
          return fail(416);
        }
        status = 206;
        res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
      }

      res.statusCode = status;
      res.setHeader('Content-Type', MIME_BY_EXTENSION[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream');
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Content-Length', size === 0 ? 0 : end - start + 1);
      if (req.method === 'HEAD' || size === 0) return res.end();

      const stream = fs.createReadStream(filePath, { start, end });
      stream.on('error', () => res.destroy());
      // The player abandons a request whenever it seeks - stop reading the file for it.
      res.on('close', () => stream.destroy());
      stream.pipe(res);
    });
  });

  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') return reject(new Error('Media server failed to start'));
      resolve({ base: `http://127.0.0.1:${address.port}/${token}`, close: () => server.close() });
    });
  });
}
