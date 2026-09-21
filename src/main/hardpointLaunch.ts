import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { shell } from 'electron';

const HARDPOINT_STATUS_URL = 'http://127.0.0.1:3921/api/status';
const HARDPOINT_UI_URL = 'http://127.0.0.1:3921/';

export type HardpointOpenResult =
  | { status: 'ok' }
  | { status: 'error'; message: string };

/** Probe from the main process — renderer fetch to :3921 is blocked by Chromium
 * (cross-origin / local-network rules) even when Hardpoint is up. */
export async function isHardpointReachable(): Promise<boolean> {
  try {
    const response = await fetch(HARDPOINT_STATUS_URL, {
      method: 'GET',
      signal: AbortSignal.timeout(2_000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/** Resolve the Hardpoint app root (sibling `hArdpoInt/`, or an installed exe). */
export function resolveHardpointRoot(): string | null {
  // KVGenius/dist/main/main → ../../../ = KVGenius; sibling hArdpoInt
  const reposRoot = path.resolve(__dirname, '../../..');
  const candidates = [
    path.join(reposRoot, '..', 'hArdpoInt'),
    path.join(reposRoot, '..', 'Hardpoint'),
    path.join(reposRoot, '..', 'hardpoint'),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, 'package.json'))) return dir;
  }

  const localAppData = process.env.LOCALAPPDATA;
  if (localAppData) {
    const installedDir = path.join(localAppData, 'Programs', 'Hardpoint');
    const exe = path.join(installedDir, 'Hardpoint.exe');
    if (fs.existsSync(exe)) return installedDir;
  }

  return null;
}

/**
 * Open Hardpoint: if its API is already up, open the UI in the default browser;
 * otherwise spawn sibling `npm run dev` (dev) or Hardpoint.exe (install).
 */
export async function openHardpoint(): Promise<HardpointOpenResult> {
  if (await isHardpointReachable()) {
    await shell.openExternal(HARDPOINT_UI_URL);
    return { status: 'ok' };
  }

  const root = resolveHardpointRoot();
  if (!root) {
    return {
      status: 'error',
      message:
        'Hardpoint not found. Clone https://github.com/gerp93/hArdpoInt next to KVGenius, or install Hardpoint.',
    };
  }

  const exe = path.join(root, 'Hardpoint.exe');
  if (fs.existsSync(exe)) {
    spawn(exe, [], { detached: true, stdio: 'ignore' }).unref();
    return { status: 'ok' };
  }

  const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  spawn(npmCmd, ['run', 'dev'], {
    cwd: root,
    detached: true,
    stdio: 'ignore',
    shell: true,
    env: process.env,
  }).unref();

  return { status: 'ok' };
}
