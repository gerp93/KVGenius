import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DownloadItemInfo } from '../shared/modelDownloads';
import { commandExists, startDownloads } from './downloadLauncher';

const items: DownloadItemInfo[] = [{ label: 'VAE', url: 'https://huggingface.co/x/resolve/main/ae.safetensors', folder: 'vae', fileName: 'ae.safetensors', destPath: '/m/vae/ae.safetensors' }];

function temp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-dlsh-'));
}

test('the script is saved (and kept, to resume) and a terminal is opened on it', async () => {
  const dir = temp();
  try {
    let launched: { command: string; args: string[] } | null = null;
    const result = await startDownloads(items, {
      scriptDir: path.join(dir, 'downloads'),
      platform: 'linux',
      isAvailable: (c) => c === 'xterm',
      launch: async (command, args) => {
        launched = { command, args };
      },
    });
    assert.equal(result.status, 'launched');
    if (result.status !== 'launched') return;
    assert.equal(result.scriptPath, path.join(dir, 'downloads', 'download-models.sh'));
    assert.match(fs.readFileSync(result.scriptPath, 'utf8'), /^#!\/bin\/sh/);
    assert.ok((fs.statSync(result.scriptPath).mode & 0o111) !== 0, 'executable');
    assert.deepEqual(launched, { command: 'xterm', args: ['-e', 'sh', result.scriptPath] });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('with no terminal, or one that will not open, the script is handed back to run by hand', async () => {
  const dir = temp();
  try {
    const none = await startDownloads(items, { scriptDir: dir, platform: 'linux', isAvailable: () => false });
    assert.equal(none.status, 'copy');
    if (none.status === 'copy') {
      assert.match(none.reason, /No terminal/);
      assert.match(none.script, /curl -L -C -/);
    }
    const broken = await startDownloads(items, {
      scriptDir: dir,
      platform: 'linux',
      isAvailable: () => true,
      launch: async () => {
        throw new Error('spawn x-terminal-emulator ENOENT');
      },
    });
    assert.equal(broken.status, 'copy');
    if (broken.status === 'copy') assert.match(broken.reason, /could not be opened.*ENOENT/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('nothing to download is an error, and an unwritable folder is reported', async () => {
  assert.equal((await startDownloads([], { scriptDir: '/tmp', platform: 'linux' })).status, 'error');
  const dir = temp();
  try {
    const blocker = path.join(dir, 'file');
    fs.writeFileSync(blocker, 'x');
    const r = await startDownloads(items, { scriptDir: path.join(blocker, 'sub'), platform: 'linux' });
    assert.equal(r.status, 'error');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('Windows and macOS scripts get their own names', async () => {
  const dir = temp();
  try {
    const win = await startDownloads(items, { scriptDir: dir, platform: 'win32', launch: async () => undefined });
    assert.equal(win.status === 'launched' && path.basename(win.scriptPath), 'download-models.ps1');
    const mac = await startDownloads(items, { scriptDir: dir, platform: 'darwin', launch: async () => undefined });
    assert.equal(mac.status === 'launched' && path.basename(mac.scriptPath), 'download-models.command');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a command is found only if it is an executable file on the PATH', () => {
  const dir = temp();
  try {
    const exe = path.join(dir, 'myterm');
    fs.writeFileSync(exe, '#!/bin/sh\n', { mode: 0o755 });
    fs.writeFileSync(path.join(dir, 'plain'), 'x', { mode: 0o644 });
    const env = { PATH: `${path.join(dir, 'nowhere')}${path.delimiter}${dir}` };
    assert.equal(commandExists('myterm', env), true);
    assert.equal(commandExists('plain', env), false);
    assert.equal(commandExists('absent', env), false);
    assert.equal(commandExists('myterm', {}), false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
