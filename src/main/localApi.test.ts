import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as http from 'node:http';
import { ApiError } from './apiService';
import { LocalApi, startLocalApi } from './localApi';
import { TOOLS } from '../shared/tools';

const TOKEN = 'secret-token';
let api: LocalApi;
const calls: Array<{ tool: string; args: unknown }> = [];

before(async () => {
  api = await startLocalApi({
    token: TOKEN,
    preferredPort: 0,
    version: '9.9.9',
    handler: async (tool, args) => {
      calls.push({ tool, args });
      if (tool === 'bad_arg') throw new ApiError('invalid_argument', 'nope');
      if (tool === 'missing') throw new ApiError('not_found', 'gone');
      if (tool === 'unfit') throw new ApiError('ffmpeg_missing', 'no ffmpeg');
      if (tool === 'boom') throw new Error('kaboom');
      if (tool === 'preview') return { data: { a: 1 }, images: [{ mimeType: 'image/jpeg', data: 'AAAA' }] };
      return { data: { tool, args } };
    },
  });
});

after(() => api.close());

function request(opts: { method?: string; path: string; headers?: Record<string, string>; body?: string; host?: string }): Promise<{ status: number; json: any }> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port: api.port, method: opts.method ?? 'GET', path: opts.path, headers: { Host: opts.host ?? `127.0.0.1:${api.port}`, ...opts.headers } },
      (res) => {
        let data = '';
        res.on('data', (c) => (data += c));
        res.on('end', () => resolve({ status: res.statusCode ?? 0, json: data ? JSON.parse(data) : null }));
      }
    );
    req.on('error', reject);
    if (opts.body !== undefined) req.write(opts.body);
    req.end();
  });
}

const auth = { Authorization: `Bearer ${TOKEN}` };
const post = (tool: string, body: unknown, headers: Record<string, string> = auth) =>
  request({ method: 'POST', path: `/v1/tools/${tool}`, headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) });

test('binds to loopback only', () => {
  assert.ok(api.port > 0);
});

test('health and tool listing need the token', async () => {
  assert.equal((await request({ path: '/v1/health' })).status, 401);
  assert.equal((await request({ path: '/v1/health', headers: { Authorization: 'Bearer wrong' } })).status, 401);
  assert.equal((await request({ path: '/v1/health', headers: { Authorization: TOKEN } })).status, 401, 'must be a Bearer token');
  const health = await request({ path: '/v1/health', headers: auth });
  assert.equal(health.status, 200);
  assert.equal(health.json.data.version, '9.9.9');
  const tools = await request({ path: '/v1/tools', headers: auth });
  assert.equal(tools.json.data.length, TOOLS.length);
});

test('requests carrying an Origin header (web pages) are refused, even with the token', async () => {
  const r = await request({ path: '/v1/health', headers: { ...auth, Origin: 'https://evil.example' } });
  assert.equal(r.status, 403);
  assert.equal(r.json.error.code, 'origin_not_allowed');
});

test('a foreign Host header (DNS rebinding) is refused', async () => {
  const r = await request({ path: '/v1/health', headers: auth, host: 'evil.example' });
  assert.equal(r.status, 403);
  assert.equal(r.json.error.code, 'host_not_allowed');
  assert.equal((await request({ path: '/v1/health', headers: auth, host: `localhost:${api.port}` })).status, 200);
});

test('tool calls pass their JSON arguments to the handler', async () => {
  const r = await post('list_jobs', { batch: 'x' });
  assert.equal(r.status, 200);
  assert.deepEqual(r.json, { ok: true, data: { tool: 'list_jobs', args: { batch: 'x' } } });
  assert.deepEqual(calls[calls.length - 1], { tool: 'list_jobs', args: { batch: 'x' } });
  assert.deepEqual((await request({ method: 'POST', path: '/v1/tools/list_jobs', headers: auth })).json.data.args, {}, 'empty body = no arguments');
});

test('previews travel alongside the data', async () => {
  const r = await post('preview', {});
  assert.deepEqual(r.json.images, [{ mimeType: 'image/jpeg', data: 'AAAA' }]);
});

test('errors map to sensible statuses', async () => {
  assert.equal((await post('bad_arg', {})).status, 400);
  assert.equal((await post('missing', {})).status, 404);
  const unfit = await post('unfit', {});
  assert.equal(unfit.status, 422);
  assert.equal(unfit.json.error.code, 'ffmpeg_missing');
  const boom = await post('boom', {});
  assert.equal(boom.status, 500);
  assert.equal(boom.json.error.message, 'kaboom');
});

test('bad JSON, unknown routes and oversized bodies are handled', async () => {
  const bad = await request({ method: 'POST', path: '/v1/tools/list_jobs', headers: auth, body: '{nope' });
  assert.equal(bad.status, 400);
  assert.equal((await request({ path: '/v2/whatever', headers: auth })).status, 404);
  assert.equal((await request({ method: 'DELETE', path: '/v1/tools/list_jobs', headers: auth })).status, 404);
  const before = calls.length;
  await request({ method: 'POST', path: '/v1/tools/list_jobs', headers: auth, body: 'x'.repeat(2 * 1024 * 1024) }).catch(() => undefined);
  assert.equal(calls.length, before, 'the oversized request never reached the handler');
});
