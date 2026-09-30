import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CallApi, NOT_RUNNING_MESSAGE, handleMessage } from './mcpProtocol';
import { ApiResponse, TOOLS } from '../shared/tools';

const ok = (data: unknown, images?: Array<{ mimeType: string; data: string }>): ApiResponse => ({ ok: true, data, images });
const send = (msg: unknown, callApi: CallApi = async () => ok({})) => handleMessage(msg, '1.2.3', callApi);

test('initialize negotiates the protocol version and advertises tools', async () => {
  const r = await send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2024-11-05' } });
  assert.equal(r?.id, 1);
  const result = r?.result as any;
  assert.equal(result.protocolVersion, '2024-11-05');
  assert.deepEqual(result.serverInfo, { name: 'kvgenius', version: '1.2.3' });
  assert.ok(result.capabilities.tools);
  const unknown = await send({ jsonrpc: '2.0', id: 2, method: 'initialize', params: { protocolVersion: '1999-01-01' } });
  assert.equal((unknown?.result as any).protocolVersion, '2025-06-18');
});

test('notifications get no reply', async () => {
  assert.equal(await send({ jsonrpc: '2.0', method: 'notifications/initialized' }), null);
  assert.equal(await send({ jsonrpc: '2.0', method: 'notifications/cancelled', params: {} }), null);
});

test('ping and tools/list', async () => {
  assert.deepEqual((await send({ jsonrpc: '2.0', id: 3, method: 'ping' }))?.result, {});
  const list = (await send({ jsonrpc: '2.0', id: 4, method: 'tools/list' }))?.result as any;
  assert.equal(list.tools.length, TOOLS.length);
  for (const tool of list.tools) {
    assert.equal(tool.inputSchema.type, 'object');
    assert.ok(tool.description.length > 20, `${tool.name} is described`);
  }
});

test('tools/call returns the data as text plus any preview as an image block', async () => {
  let seen: { tool: string; args: unknown } | null = null;
  const r = await send({ jsonrpc: '2.0', id: 5, method: 'tools/call', params: { name: 'get_item', arguments: { item_id: 'gen-1' } } }, async (tool, args) => {
    seen = { tool, args };
    return ok({ hello: 'world' }, [{ mimeType: 'image/jpeg', data: 'QUJD' }]);
  });
  assert.deepEqual(seen, { tool: 'get_item', args: { item_id: 'gen-1' } });
  const result = r?.result as any;
  assert.equal(result.isError, undefined);
  assert.deepEqual(JSON.parse(result.content[0].text), { hello: 'world' });
  assert.deepEqual(result.content[1], { type: 'image', data: 'QUJD', mimeType: 'image/jpeg' });
});

test('an API error becomes an isError tool result the model can read', async () => {
  const r = await send({ jsonrpc: '2.0', id: 6, method: 'tools/call', params: { name: 'get_job', arguments: {} } }, async () => ({
    ok: false,
    error: { code: 'not_found', message: 'No job 3.' },
  }));
  const result = r?.result as any;
  assert.equal(result.isError, true);
  assert.match(result.content[0].text, /No job 3\. \(not_found\)/);
});

test('when the app cannot be reached the result explains how to fix it', async () => {
  const r = await send({ jsonrpc: '2.0', id: 7, method: 'tools/call', params: { name: 'list_jobs' } }, async () => {
    throw new Error('ECONNREFUSED');
  });
  const result = r?.result as any;
  assert.equal(result.isError, true);
  assert.equal(result.content[0].text, NOT_RUNNING_MESSAGE);
});

test('unknown tools and methods are protocol errors', async () => {
  assert.equal((await send({ jsonrpc: '2.0', id: 8, method: 'tools/call', params: { name: 'nope' } }))?.error?.code, -32602);
  assert.equal((await send({ jsonrpc: '2.0', id: 9, method: 'resources/list' }))?.error?.code, -32601);
  assert.equal((await send({ jsonrpc: '2.0', id: 10, method: 'tools/call', params: {} }))?.error?.code, -32602);
});
