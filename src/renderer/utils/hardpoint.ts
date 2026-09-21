export const HARDPOINT_API_BASE = 'http://127.0.0.1:3921';

/** True when Hardpoint's loopback API answers. Probes via main-process IPC —
 * a renderer `fetch` to :3921 fails under Chromium even when Hardpoint is up. */
export async function isHardpointReachable(): Promise<boolean> {
  try {
    return await window.kvgenius.hardpointIsReachable();
  } catch {
    return false;
  }
}
