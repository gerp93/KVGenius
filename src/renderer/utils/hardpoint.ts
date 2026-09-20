export const HARDPOINT_API_BASE = 'http://127.0.0.1:3921';

/** True when Hardpoint's loopback API answers. */
export async function isHardpointReachable(): Promise<boolean> {
  try {
    const response = await fetch(`${HARDPOINT_API_BASE}/api/status`, {
      method: 'GET',
      signal: AbortSignal.timeout(2_000),
    });
    return response.ok;
  } catch {
    return false;
  }
}
