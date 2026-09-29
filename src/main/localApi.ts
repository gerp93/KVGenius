import * as crypto from 'crypto';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import { TOOLS, ApiResponse, ToolResult } from '../shared/tools';
import { ApiError } from './apiService';

/** Where the running app tells clients how to reach it (read by the MCP stdio shim). */
export interface DiscoveryInfo {
  port: number;
  token: string;
  pid: number;
  version: string;
}

export interface LocalApiOptions {
  token: string;
  preferredPort: number;
  version: string;
  handler: (tool: string, args: unknown) => Promise<ToolResult>;
}

export interface LocalApi {
  port: number;
  close: () => Promise<void>;
}

const MAX_BODY_BYTES = 1024 * 1024;

const STATUS_FOR_CODE: Record<string, number> = {
  invalid_argument: 400,
  not_found: 404,
  unknown_tool: 404,
};

function send(res: http.ServerResponse, status: number, body: ApiResponse | Record<string, unknown>): void {
  const json = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(json), 'Cache-Control': 'no-store' });
  res.end(json);
}

const errorBody = (code: string, message: string): ApiResponse => ({ ok: false, error: { code, message } });

function tokenMatches(header: string | undefined, token: string): boolean {
  const match = /^Bearer (.+)$/.exec(header ?? '');
  if (!match) return false;
  const a = crypto.createHash('sha256').update(match[1]).digest();
  const b = crypto.createHash('sha256').update(token).digest();
  return crypto.timingSafeEqual(a, b);
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new ApiError('invalid_argument', 'Request body is too large.'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

/**
 * The app's local control API: `GET /v1/health`, `GET /v1/tools`, `POST /v1/tools/<name>` with the
 * tool's arguments as a JSON body. It is what the MCP stdio shim talks to, and any other local
 * client can use it too. Loopback only, and every request needs the bearer token. Requests that
 * carry an Origin header (i.e. from a web page) or a Host that is not loopback are refused, so a
 * web page cannot drive it even by DNS rebinding.
 */
export function startLocalApi(options: LocalApiOptions): Promise<LocalApi> {
  const server = http.createServer(async (req, res) => {
    try {
      const port = (server.address() as { port: number }).port;
      if (req.headers.origin !== undefined) {
        return send(res, 403, errorBody('origin_not_allowed', 'Browser requests are not accepted.'));
      }
      const allowedHosts = [`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`];
      if (!allowedHosts.includes(req.headers.host ?? '')) {
        return send(res, 403, errorBody('host_not_allowed', 'Unexpected Host header.'));
      }
      if (!tokenMatches(req.headers.authorization, options.token)) {
        return send(res, 401, errorBody('unauthorized', 'Missing or wrong bearer token.'));
      }

      const url = new URL(req.url ?? '/', 'http://localhost');
      if (req.method === 'GET' && url.pathname === '/v1/health') {
        return send(res, 200, { ok: true, data: { app: 'kvgenius', version: options.version } });
      }
      if (req.method === 'GET' && url.pathname === '/v1/tools') {
        return send(res, 200, { ok: true, data: TOOLS });
      }
      const toolMatch = /^\/v1\/tools\/([a-z_]+)$/.exec(url.pathname);
      if (req.method === 'POST' && toolMatch) {
        const raw = await readBody(req);
        let args: unknown = {};
        if (raw.trim() !== '') {
          try {
            args = JSON.parse(raw);
          } catch {
            return send(res, 400, errorBody('invalid_argument', 'Request body is not valid JSON.'));
          }
        }
        const result = await options.handler(toolMatch[1], args);
        return send(res, 200, { ok: true, ...result });
      }
      return send(res, 404, errorBody('not_found', 'No such endpoint.'));
    } catch (err) {
      if (err instanceof ApiError) return send(res, STATUS_FOR_CODE[err.code] ?? 422, errorBody(err.code, err.message));
      return send(res, 500, errorBody('error', err instanceof Error ? err.message : String(err)));
    }
  });

  return new Promise((resolve, reject) => {
    const listen = (port: number) => {
      server.once('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EADDRINUSE' && port !== 0) listen(0);
        else reject(err);
      });
      server.listen(port, '127.0.0.1', () => {
        server.removeAllListeners('error');
        resolve({
          port: (server.address() as { port: number }).port,
          close: () =>
            new Promise<void>((done) => {
              server.close(() => done());
              server.closeAllConnections();
            }),
        });
      });
    };
    listen(options.preferredPort);
  });
}

export function newApiToken(): string {
  return crypto.randomBytes(32).toString('hex');
}

export function writeDiscoveryFile(file: string, info: DiscoveryInfo): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(info, null, 2), { mode: 0o600 });
  try {
    fs.chmodSync(file, 0o600);
  } catch {
    // Not supported on every filesystem (e.g. Windows); the file is in the user's own profile anyway.
  }
}

export function removeDiscoveryFile(file: string): void {
  fs.rmSync(file, { force: true });
}
