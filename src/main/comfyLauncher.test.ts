import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { detectComfyUIProgram, launchComfyUIProgram } from './comfyLauncher';

const never = async () => {
  throw new Error('openApp should not be used');
};

test('detects ComfyUI Desktop in its default Windows install folder', () => {
  const exe = path.win32.join('C:\\Users\\k\\AppData\\Local', 'Programs', 'ComfyUI', 'ComfyUI.exe');
  assert.equal(
    detectComfyUIProgram({ platform: 'win32', env: { LOCALAPPDATA: 'C:\\Users\\k\\AppData\\Local' }, exists: (p) => p === exe }),
    exe
  );
  assert.equal(detectComfyUIProgram({ platform: 'win32', env: {}, exists: () => true }), null, 'no env, nothing to look in');
});

test('detects the macOS app in /Applications, then ~/Applications', () => {
  assert.equal(detectComfyUIProgram({ platform: 'darwin', home: '/Users/k', exists: (p) => p === '/Applications/ComfyUI.app' }), '/Applications/ComfyUI.app');
  assert.equal(detectComfyUIProgram({ platform: 'darwin', home: '/Users/k', exists: (p) => p === '/Users/k/Applications/ComfyUI.app' }), '/Users/k/Applications/ComfyUI.app');
});

test('nothing is guessed on Linux, or when nothing is installed', () => {
  assert.equal(detectComfyUIProgram({ platform: 'linux', exists: () => true }), null);
  assert.equal(detectComfyUIProgram({ platform: 'darwin', exists: () => false }), null);
});

test('a missing program is reported, not thrown', async () => {
  const r = await launchComfyUIProgram(path.join(os.tmpdir(), 'kvg-no-such-comfy.sh'), never);
  assert.equal(r.status, 'error');
  assert.match((r as { message: string }).message, /Not found/);
});

test('a .app bundle is opened through openApp, and its error is passed on', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-launch-'));
  const app = path.join(dir, 'ComfyUI.app');
  fs.mkdirSync(app);
  try {
    let opened = '';
    assert.deepEqual(await launchComfyUIProgram(app, async (p) => ((opened = p), '')), { status: 'launched', path: app });
    assert.equal(opened, app);
    assert.deepEqual(await launchComfyUIProgram(app, async () => 'nope'), { status: 'error', message: 'nope' });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a script is started detached in its own folder', { skip: process.platform === 'win32' }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-launch-'));
  const script = path.join(dir, 'run.sh');
  fs.writeFileSync(script, 'pwd > started.txt\n'); // no exec bit on purpose
  try {
    assert.deepEqual(await launchComfyUIProgram(script, never), { status: 'launched', path: script });
    const marker = path.join(dir, 'started.txt');
    for (let i = 0; i < 50 && !fs.existsSync(marker); i++) await new Promise((r) => setTimeout(r, 50));
    assert.equal(fs.readFileSync(marker, 'utf-8').trim(), fs.realpathSync(dir), 'ran with the program folder as its working directory');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a program that cannot be executed is reported', { skip: process.platform === 'win32' }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-launch-'));
  const file = path.join(dir, 'ComfyUI.AppImage');
  fs.writeFileSync(file, 'not executable');
  try {
    const r = await launchComfyUIProgram(file, never);
    assert.equal(r.status, 'error');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
