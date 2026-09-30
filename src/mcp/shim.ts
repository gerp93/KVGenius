import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { ApiResponse } from '../shared/tools';
import { handleMessage } from './mcpProtocol';

/**
 * MCP stdio server for KVGenius. An MCP client (e.g. Claude Desktop) launches this as a subprocess;
 * it holds no state of its own and forwards every tool call to the running KVGenius app's local API
 * (see src/main/localApi.ts), whose address and token it reads from the discovery file the app
 * writes while the API is enabled. Runs under plain Node or under the app's own Electron binary
 * with ELECTRON_RUN_AS_NODE=1.
 */
const VERSION = process.env.KVGENIUS_VERSION ?? '0.0.0';

function defaultDiscoveryFile(): string {
  const home = os.homedir();
  const base =
    process.platform === 'win32'
      ? (process.env.APPDATA ?? path.join(home, 'AppData', 'Roaming'))
      : process.platform === 'darwin'
        ? path.join(home, 'Library', 'Application Support')
        : (process.env.XDG_CONFIG_HOME ?? path.join(home, '.config'));
  return path.join(base, 'kvgenius', 'mcp-api.json');
}

async function callApi(tool: string, args: unknown): Promise<ApiResponse> {
  const file = process.env.KVGENIUS_API_FILE || defaultDiscoveryFile();
  const info = JSON.parse(fs.readFileSync(file, 'utf-8')) as { port: number; token: string };
  const resp = await fetch(`http://127.0.0.1:${info.port}/v1/tools/${tool}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${info.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(args ?? {}),
  });
  return (await resp.json()) as ApiResponse;
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
let pending = 0;
let closed = false;

function maybeExit(): void {
  if (closed && pending === 0) process.exit(0);
}

rl.on('line', (line) => {
  if (line.trim() === '') return;
  pending++;
  void (async () => {
    try {
      const reply = await handleMessage(JSON.parse(line), VERSION, callApi);
      if (reply) process.stdout.write(JSON.stringify(reply) + '\n');
    } catch (err) {
      process.stderr.write(`kvgenius-mcp: could not handle message: ${String(err)}\n`);
    } finally {
      pending--;
      maybeExit();
    }
  })();
});
rl.on('close', () => {
  closed = true;
  maybeExit();
});
