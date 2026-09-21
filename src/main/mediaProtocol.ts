import * as fs from 'fs';
import * as path from 'path';
import { Readable } from 'stream';

export const MEDIA_SCHEME = 'kvimage';

/** Privileges for protocol.registerSchemesAsPrivileged() - `stream` is what lets <video>/<audio>
 * elements read the scheme. Exported so the standalone playback test registers it identically. */
export const MEDIA_SCHEME_PRIVILEGES = { secure: true, supportFetchAPI: true, corsEnabled: true, stream: true };

export const MIME_BY_EXTENSION: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
};

/**
 * Serves a generated image/video for a `kvimage://` request. Done by hand (not net.fetch of a
 * file:// URL) because a <video> element only plays - and only seeks - if the response has a
 * real video Content-Type, advertises Accept-Ranges, and answers `Range:` requests with 206.
 */
/** True if `resolvedPath` is inside one of the allowed directories or is an individually allowed file. */
export function isAllowedMediaPath(
  resolvedPath: string,
  allowedDirectories: string[],
  extraAllowedFiles: ReadonlySet<string>
): boolean {
  const allowedDirs = allowedDirectories.map((dir) => path.resolve(dir));
  return allowedDirs.some((dir) => resolvedPath.startsWith(dir + path.sep)) || extraAllowedFiles.has(resolvedPath);
}

export const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.mkv'];

export async function handleMediaRequest(
  request: Request,
  allowedDirectories: string[],
  extraAllowedFiles: ReadonlySet<string> = new Set()
): Promise<Response> {
  const encodedPath = request.url.replace('kvimage://', '').replace(/[?#].*$/, '');
  const filePath = decodeURIComponent(encodedPath);
  // Only ever serve files inside the app's own output directories (plus the individual source
  // images the user picked in a file dialog) - the renderer passes
  // paths back that originated from the database, but this is cheap insurance against a
  // malformed/crafted kvimage:// URL reaching outside it.
  const resolved = path.resolve(filePath);
  if (!isAllowedMediaPath(resolved, allowedDirectories, extraAllowedFiles)) {
    return new Response('Forbidden', { status: 403 });
  }

  let size: number;
  try {
    size = (await fs.promises.stat(resolved)).size;
  } catch {
    return new Response('Not found', { status: 404 });
  }

  const headers: Record<string, string> = {
    'Content-Type': MIME_BY_EXTENSION[path.extname(resolved).toLowerCase()] ?? 'application/octet-stream',
    'Accept-Ranges': 'bytes',
  };
  if (size === 0) return new Response(null, { status: 200, headers });

  let start = 0;
  let end = size - 1;
  let status = 200;
  const match = /^bytes=(\d*)-(\d*)$/.exec(request.headers.get('range') ?? '');
  if (match && (match[1] !== '' || match[2] !== '')) {
    if (match[1] === '') {
      start = Math.max(0, size - Number(match[2]));
    } else {
      start = Number(match[1]);
      if (match[2] !== '') end = Math.min(Number(match[2]), size - 1);
    }
    if (start > end) {
      return new Response(null, { status: 416, headers: { ...headers, 'Content-Range': `bytes */${size}` } });
    }
    status = 206;
    headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
  }
  headers['Content-Length'] = String(end - start + 1);

  const body = Readable.toWeb(fs.createReadStream(resolved, { start, end })) as ReadableStream;
  return new Response(body, { status, headers });
}
