import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { DownloadItemInfo, DownloadStartResult } from '../shared/modelDownloads';
import { buildScript, scriptFileName, terminalLaunch } from './downloadScript';

/** True when `command` is an executable on the PATH (how Linux terminals are found). */
export function commandExists(command: string, env: NodeJS.ProcessEnv = process.env): boolean {
  for (const dir of (env.PATH ?? '').split(path.delimiter)) {
    if (!dir) continue;
    try {
      fs.accessSync(path.join(dir, command), fs.constants.X_OK);
      return true;
    } catch {
      // not in this folder
    }
  }
  return false;
}

/** Starts a program on its own (it keeps running if KVGenius closes). Resolves once it has started, rejects if it cannot. */
export function spawnDetached(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { detached: true, stdio: 'ignore' });
    child.once('error', reject);
    child.once('spawn', () => {
      child.unref();
      resolve();
    });
  });
}

export interface StartOptions {
  /** Where the script is saved: kept, so running it again later resumes an interrupted download. */
  scriptDir: string;
  platform?: NodeJS.Platform;
  isAvailable?: (command: string) => boolean;
  launch?: (command: string, args: string[]) => Promise<void>;
}

/**
 * Saves the download script and opens a terminal window running it, so the terminal shows the progress and a
 * re-run resumes. When no terminal can be opened the script is handed back to be run by hand - never a failure
 * the user is stuck with.
 */
export async function startDownloads(items: DownloadItemInfo[], options: StartOptions): Promise<DownloadStartResult> {
  const platform = options.platform ?? process.platform;
  if (items.length === 0) return { status: 'error', message: 'There is nothing to download.' };
  const script = buildScript(platform, items);
  const scriptPath = path.join(options.scriptDir, scriptFileName(platform));
  try {
    fs.mkdirSync(options.scriptDir, { recursive: true });
    fs.writeFileSync(scriptPath, script, { mode: 0o755 });
    if (platform !== 'win32') fs.chmodSync(scriptPath, 0o755);
  } catch (err) {
    return { status: 'error', message: `The download script could not be saved: ${err instanceof Error ? err.message : String(err)}` };
  }
  const command = terminalLaunch(platform, scriptPath, options.isAvailable ?? commandExists);
  if (!command) {
    return { status: 'copy', scriptPath, script, reason: 'No terminal program was found to open it in.' };
  }
  try {
    await (options.launch ?? spawnDetached)(command.command, command.args);
    return { status: 'launched', scriptPath };
  } catch (err) {
    return { status: 'copy', scriptPath, script, reason: `The terminal could not be opened (${err instanceof Error ? err.message : String(err)}).` };
  }
}
