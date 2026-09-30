import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export type LaunchResult = { status: 'launched'; path: string } | { status: 'error'; message: string };

interface DetectOptions {
  platform?: NodeJS.Platform;
  env?: NodeJS.ProcessEnv;
  home?: string;
  exists?: (p: string) => boolean;
}

/**
 * Where ComfyUI Desktop installs itself by default. Only the Desktop app is guessed at - the
 * standalone/portable ComfyUI (a folder with run_*.bat / a venv) and Linux AppImages live wherever
 * the user put them, so those come from the path chosen in Settings.
 */
export function detectComfyUIProgram(options: DetectOptions = {}): string | null {
  const platform = options.platform ?? process.platform;
  const env = options.env ?? process.env;
  const home = options.home ?? os.homedir();
  const exists = options.exists ?? fs.existsSync;

  const candidates: string[] = [];
  if (platform === 'win32') {
    if (env.LOCALAPPDATA) candidates.push(path.win32.join(env.LOCALAPPDATA, 'Programs', 'ComfyUI', 'ComfyUI.exe'));
    if (env.ProgramFiles) candidates.push(path.win32.join(env.ProgramFiles, 'ComfyUI', 'ComfyUI.exe'));
  } else if (platform === 'darwin') {
    candidates.push('/Applications/ComfyUI.app', path.posix.join(home, 'Applications', 'ComfyUI.app'));
  }
  return candidates.find((c) => exists(c)) ?? null;
}

/**
 * Starts a ComfyUI program and lets it run on its own (it keeps running when KVGenius closes).
 * `openApp` opens a macOS .app bundle; it resolves to an error message, or '' on success.
 * Resolves once the program has actually started or failed to - not once ComfyUI is ready to serve.
 */
export function launchComfyUIProgram(target: string, openApp: (appPath: string) => Promise<string>): Promise<LaunchResult> {
  if (!fs.existsSync(target)) {
    return Promise.resolve({ status: 'error', message: `Not found: ${target}. Choose the ComfyUI program again in Settings.` });
  }
  const ext = path.extname(target).toLowerCase();
  if (ext === '.app') {
    return openApp(target).then((error) => (error ? { status: 'error', message: error } : { status: 'launched', path: target }));
  }

  // Batch files need a shell on Windows; shell scripts are run through sh so they need no exec bit.
  const windowsScript = process.platform === 'win32' && (ext === '.bat' || ext === '.cmd');
  const command = ext === '.sh' && process.platform !== 'win32' ? '/bin/sh' : target;
  const args = ext === '.sh' && process.platform !== 'win32' ? [target] : [];

  return new Promise((resolve) => {
    let settled = false;
    const finish = (result: LaunchResult) => {
      if (!settled) {
        settled = true;
        resolve(result);
      }
    };
    try {
      const child = spawn(windowsScript ? `"${target}"` : command, args, {
        cwd: path.dirname(target),
        detached: true,
        stdio: 'ignore',
        shell: windowsScript,
      });
      child.once('error', (err) => finish({ status: 'error', message: `Could not start ${path.basename(target)}: ${err.message}` }));
      child.once('spawn', () => {
        child.unref();
        finish({ status: 'launched', path: target });
      });
    } catch (err) {
      finish({ status: 'error', message: `Could not start ${path.basename(target)}: ${err instanceof Error ? err.message : String(err)}` });
    }
  });
}
