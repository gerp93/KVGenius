import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFile, execFileSync } from 'node:child_process';
import * as fs from 'fs';
import * as http from 'http';
import * as os from 'os';
import * as path from 'path';
import { MODEL_MANIFEST, manifestFeature } from '../shared/modelManifest';
import { emptyInstalled, ModelStatusReport } from '../shared/modelStatus';
import { DownloadItemInfo } from '../shared/modelDownloads';
import { buildPosixScript, buildPowershellScript, planDownloads, powershellQuote, scriptFileName, shellQuote, terminalLaunch } from './downloadScript';

const zImage = manifestFeature('z-image')!;

function report(installed: Partial<ReturnType<typeof emptyInstalled>>, source: ModelStatusReport['source'] = 'comfyui'): ModelStatusReport {
  return { source, installed: { ...emptyInstalled(), ...installed } };
}

test('every manifest file with fixed files has a Hugging Face link that matches its name and folder', () => {
  for (const feature of MODEL_MANIFEST) {
    for (const file of feature.files) {
      assert.ok(file.url, `${file.file} has a link`);
      const url = new URL(file.url as string);
      assert.equal(url.protocol, 'https:');
      assert.equal(url.hostname, 'huggingface.co');
      assert.ok(url.pathname.endsWith(`/split_files/${file.folder}/${file.file}`), file.url);
    }
  }
});

test('only the missing files are planned, into their own folders; subfolder files are reported, not fetched again', () => {
  const modelsDir = path.join(os.tmpdir(), 'ComfyUI', 'models');
  const plan = planDownloads([zImage], report({ diffusion_models: ['z_image_turbo_bf16.safetensors'], text_encoders: ['extra/qwen_3_4b.safetensors'] }), modelsDir);
  assert.deepEqual(plan.items.map((i) => i.fileName), ['ae.safetensors']);
  assert.equal(plan.items[0].destPath, path.join(modelsDir, 'vae', 'ae.safetensors'));
  assert.deepEqual(plan.inSubfolder, ['qwen_3_4b.safetensors']);
  assert.equal(planDownloads([zImage], report({}), modelsDir).items.length, 3);
  assert.equal(planDownloads([zImage], report({ diffusion_models: ['z_image_turbo_bf16.safetensors'], text_encoders: ['qwen_3_4b.safetensors'], vae: ['ae.safetensors'] }), modelsDir).items.length, 0);
});

test('with nothing known, nothing is planned, and a feature with no fixed files plans nothing', () => {
  assert.equal(planDownloads([zImage], report({}, 'none'), '/m').items.length, 0);
  assert.equal(planDownloads([manifestFeature('upscale')!], report({}), '/m').items.length, 0);
});

test('quoting leaves nothing to expand or run', () => {
  assert.equal(shellQuote("it's"), "'it'\\''s'");
  assert.equal(powershellQuote("it's"), "'it''s'");
  const hostile = `/tmp/a b/$(touch pwned)\`id\`;"x"&|<>%'q`;
  assert.equal(execFileSync('sh', ['-c', `printf %s ${shellQuote(hostile)}`], { encoding: 'utf8' }), hostile);
});

function runScript(script: string): Promise<{ stdout: string; code: number }> {
  return new Promise((resolve) => {
    const child = execFile('sh', [script], { encoding: 'utf8' }, (err, stdout) => resolve({ stdout, code: err ? Number((err as { code?: unknown }).code) : 0 }));
    child.stdin?.end();
  });
}

const hostileDir = fs.mkdtempSync(path.join(os.tmpdir(), "kvg dl $(touch PWNED) `id` ;&|' "));
test('the POSIX script downloads, resumes past finished files, reports failures, and runs nothing it was not given', async () => {
  const body = Buffer.from('model-bytes'.repeat(1000));
  const server = http.createServer((req, res) => {
    if (req.url === '/missing.safetensors') {
      res.statusCode = 404;
      res.end();
      return;
    }
    res.setHeader('Content-Length', body.length);
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as { port: number }).port;
  try {
    const items: DownloadItemInfo[] = [
      { label: 'Good file', url: `http://127.0.0.1:${port}/good.safetensors`, folder: 'vae', fileName: "o'k.safetensors", destPath: path.join(hostileDir, 'vae', "o'k.safetensors") },
      { label: 'Gone file', url: `http://127.0.0.1:${port}/missing.safetensors`, folder: 'vae', fileName: 'gone.safetensors', destPath: path.join(hostileDir, 'vae', 'gone.safetensors') },
    ];
    const script = path.join(hostileDir, 'download.sh');
    fs.writeFileSync(script, buildPosixScript(items));
    // The script talks to a server in this same process, so it runs without blocking the event loop; its closing
    // 'Press Enter' prompt reads stdin, which is closed so the read ends at once.
    const first = await runScript(script);
    const output = first.stdout;
    assert.equal(first.code, 1, 'a file that did not finish makes the script exit non-zero');
    assert.match(output, /Downloading Good file/);
    assert.match(output, /FAILED/);
    assert.match(output, /1 file\(s\) did not finish/);
    assert.deepEqual(fs.readFileSync(items[0].destPath), body);
    assert.equal(fs.existsSync(`${items[0].destPath}.part`), false, 'a finished file leaves no .part behind');
    assert.equal(fs.existsSync(items[1].destPath), false);
    assert.equal(fs.existsSync(path.join(process.cwd(), 'PWNED')) || fs.existsSync(path.join(hostileDir, 'PWNED')), false);
    assert.equal(fs.readdirSync(hostileDir).some((f) => f.includes('PWNED') && !f.startsWith('kvg')), false);
    // a second run sees the finished file and does not fetch it again
    const again = await runScript(script);
    assert.match(again.stdout, /Already there/);
  } finally {
    server.close();
    fs.rmSync(hostileDir, { recursive: true, force: true });
  }
});

test('the POSIX script is valid shell for any item list', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-sh-'));
  try {
    for (const items of [[], [{ label: "It's", url: 'https://x/y', folder: 'vae', fileName: 'a', destPath: "/a b/it's" }]]) {
      const file = path.join(dir, 'script.sh');
      fs.writeFileSync(file, buildPosixScript(items));
      execFileSync('sh', ['-n', file]);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('the PowerShell script keeps every value in single-quoted literals', () => {
  const item: DownloadItemInfo = { label: "It's $(bad)", url: 'https://x/y', folder: 'vae', fileName: 'a', destPath: "C:\\Users\\a b\\it's $env:X\\a.safetensors" };
  const script = buildPowershellScript([item]);
  assert.ok(script.includes(`Get-OneFile 'https://x/y' 'C:\\Users\\a b\\it''s $env:X\\a.safetensors' 'It''s $(bad)'`));
  // the only double-quoted strings are the script's own fixed text and variables, never user values
  assert.equal(script.includes('$(bad)"'), false);
  assert.match(script, /Read-Host 'Press Enter to close'/);
});

test('each platform gets its own script name and terminal command; the script path is always a separate argument', () => {
  assert.equal(scriptFileName('win32'), 'download-models.ps1');
  assert.equal(scriptFileName('darwin'), 'download-models.command');
  assert.equal(scriptFileName('linux'), 'download-models.sh');
  const script = '/home/a b/download models.sh';
  assert.deepEqual(terminalLaunch('darwin', script, () => false), { command: 'open', args: ['-a', 'Terminal', script] });
  const win = terminalLaunch('win32', 'C:\\x y\\d.ps1', () => false);
  assert.equal(win?.command, 'cmd.exe');
  assert.equal(win?.args[win.args.length - 1], 'C:\\x y\\d.ps1');
  assert.deepEqual(terminalLaunch('linux', script, (c) => c === 'konsole'), { command: 'konsole', args: ['-e', 'sh', script] });
  assert.deepEqual(terminalLaunch('linux', script, (c) => c === 'gnome-terminal' || c === 'xterm'), { command: 'gnome-terminal', args: ['--', 'sh', script] });
  assert.equal(terminalLaunch('linux', script, () => false), null);
});
