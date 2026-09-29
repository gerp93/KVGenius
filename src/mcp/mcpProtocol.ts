import { ApiResponse, TOOLS } from '../shared/tools';

/**
 * The MCP side of the stdio shim: newline-delimited JSON-RPC 2.0, handling the handful of methods
 * a tools-only server needs. Kept free of I/O (the caller supplies `callApi`) so it can be tested.
 * Nothing here is specific to one client - any MCP client speaking stdio can use it.
 */
export type CallApi = (tool: string, args: unknown) => Promise<ApiResponse>;

interface JsonRpcRequest {
  jsonrpc?: string;
  id?: number | string | null;
  method?: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcReply {
  jsonrpc: '2.0';
  id: number | string | null;
  result?: unknown;
  error?: { code: number; message: string };
}

const SUPPORTED_PROTOCOLS = ['2025-06-18', '2025-03-26', '2024-11-05'];

export const NOT_RUNNING_MESSAGE =
  'KVGenius is not reachable. Make sure the KVGenius app is open and "Allow other apps to control KVGenius (MCP)" is turned on in its Settings.';

type ContentBlock = { type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string };

export function toolResultContent(response: ApiResponse): { content: ContentBlock[]; isError?: boolean } {
  if (!response.ok) {
    return { content: [{ type: 'text', text: `${response.error.message} (${response.error.code})` }], isError: true };
  }
  const content: ContentBlock[] = [{ type: 'text', text: JSON.stringify(response.data, null, 2) }];
  for (const image of response.images ?? []) content.push({ type: 'image', data: image.data, mimeType: image.mimeType });
  return { content };
}

/** Returns the reply to send, or null for notifications (which get none). */
export async function handleMessage(message: unknown, serverVersion: string, callApi: CallApi): Promise<JsonRpcReply | null> {
  const req = message as JsonRpcRequest;
  const isNotification = req.id === undefined || req.id === null;
  const reply = (result: unknown): JsonRpcReply | null => (isNotification ? null : { jsonrpc: '2.0', id: req.id as number | string, result });
  const error = (code: number, text: string): JsonRpcReply | null =>
    isNotification ? null : { jsonrpc: '2.0', id: req.id as number | string, error: { code, message: text } };

  switch (req.method) {
    case 'initialize': {
      const asked = typeof req.params?.protocolVersion === 'string' ? req.params.protocolVersion : '';
      return reply({
        protocolVersion: SUPPORTED_PROTOCOLS.includes(asked) ? asked : SUPPORTED_PROTOCOLS[0],
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: 'kvgenius', version: serverVersion },
        instructions:
          'KVGenius generates images and videos through a local ComfyUI. Call list_capabilities first. Generation jobs are queued and run one at a time: submit them, then poll list_jobs / get_job. Reference files by library item id.',
      });
    }
    case 'ping':
      return reply({});
    case 'tools/list':
      return reply({ tools: TOOLS });
    case 'tools/call': {
      const name = req.params?.name;
      if (typeof name !== 'string') return error(-32602, 'tools/call needs a tool name.');
      if (!TOOLS.some((t) => t.name === name)) return error(-32602, `Unknown tool: ${name}`);
      try {
        return reply(toolResultContent(await callApi(name, req.params?.arguments ?? {})));
      } catch {
        return reply({ content: [{ type: 'text', text: NOT_RUNNING_MESSAGE }], isError: true });
      }
    }
    default:
      if (req.method?.startsWith('notifications/')) return null;
      return error(-32601, `Method not found: ${req.method}`);
  }
}
