import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { startLocalApi, writeDiscoveryFile } from '../main/localApi';

/** Runs the real compiled shim as a subprocess, the way an MCP client launches it. */
function launchShim(discoveryFile: string) {
  const child = spawn(process.execPath, [path.join(__dirname, 'shim.js')], {
    env: { ...process.env, KVGENIUS_API_FILE: discoveryFile, KVGENIUS_VERSION: '7.7.7' },
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const lines = readline.createInterface({ input: child.stdout });
  const waiting = new Map<number, (reply: any) => void>();
  let exited = false;
  const failWaiting = new Error('the shim exited before replying');
  const rejecters = new Map<number, (err: Error) => void>();
  child.on('exit', () => {
    exited = true;
    rejecters.forEach((reject) => reject(failWaiting));
  });
  lines.on('line', (line) => {
    const reply = JSON.parse(line);
    waiting.get(reply.id)?.(reply);
  });
  let nextId = 1;
  return {
    child,
    request(method: string, params?: unknown): Promise<any> {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        if (exited) return reject(failWaiting);
        waiting.set(id, resolve);
        rejecters.set(id, reject);
        child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
      });
    },
    notify(method: string) {
      child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method }) + '\n');
    },
    close: () =>
      new Promise<number | null>((resolve) => {
        if (exited) return resolve(child.exitCode);
        child.on('exit', (code) => resolve(code));
        child.stdin.end();
      }),
  };
}

test('the stdio shim speaks MCP and forwards tool calls to the running app', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-shim-'));
  const file = path.join(dir, 'mcp-api.json');
  const seen: Array<{ tool: string; args: unknown }> = [];
  const api = await startLocalApi({
    token: 't0ken',
    preferredPort: 0,
    version: '1',
    handler: async (tool, args) => {
      seen.push({ tool, args });
      return { data: { echoed: args }, images: tool === 'get_item' ? [{ mimeType: 'image/jpeg', data: 'QUJD' }] : undefined };
    },
  });
  writeDiscoveryFile(file, { port: api.port, token: 't0ken', pid: process.pid, version: '1' });
  const shim = launchShim(file);
  try {
    const init = await shim.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
    assert.equal(init.result.serverInfo.version, '7.7.7');
    shim.notify('notifications/initialized');

    const tools = await shim.request('tools/list');
    assert.ok(tools.result.tools.some((t: any) => t.name === 'generate_video'));

    const call = await shim.request('tools/call', { name: 'list_jobs', arguments: { batch: 'mv' } });
    assert.deepEqual(JSON.parse(call.result.content[0].text), { echoed: { batch: 'mv' } });
    assert.deepEqual(seen, [{ tool: 'list_jobs', args: { batch: 'mv' } }]);

    const withImage = await shim.request('tools/call', { name: 'get_item', arguments: { item_id: 'gen-1' } });
    assert.equal(withImage.result.content[1].type, 'image');

    // The app going away is reported as a readable tool error, not a crash.
    await api.close();
    const down = await shim.request('tools/call', { name: 'list_jobs', arguments: {} });
    assert.equal(down.result.isError, true);
    assert.match(down.result.content[0].text, /KVGenius is not reachable/);

    // And a missing discovery file (app never started / API disabled) is the same message.
    fs.rmSync(file);
    const noFile = await shim.request('tools/call', { name: 'list_jobs', arguments: {} });
    assert.equal(noFile.result.isError, true);
  } finally {
    assert.equal(await shim.close(), 0, 'exits cleanly when the client closes stdin');
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
