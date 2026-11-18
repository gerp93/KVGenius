import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { initDatabase, insertGeneration } from './db';
import { saveStyle } from './styles';
import { JobQueue, JobRunner } from './jobQueue';
import { AssemblyManager } from './assembly';
import { ApiError, ApiService, MAX_IMAGE_FILE_BYTES } from './apiService';
import { startLocalApi, LocalApi } from './localApi';
import { createWorkspace, listWorkspaces, styleExamplesWorkspaceId, updateWorkspace } from './workspaces';
import { Workspace } from '../shared/workspaces';
import { TOOLS } from '../shared/tools';

/**
 * The API's optional `workspace` argument, `get_image_file`, and z-image's 16-multiple sizes. Kept apart from
 * apiService.test.ts (whose tests share state and depend on their order) with its own database and queue.
 */
let dir: string;
let db: DatabaseSync;
let service: ApiService;
let queue: JobQueue;
let currentWorkspace = 1;
let released: Array<() => void> = [];
let forge: Workspace;
let secret: Workspace;
const written = new Map<number, Buffer>();

/** Like the real runner: the result is recorded in the workspace the job was queued in. */
const fakeRunner: JobRunner = (job) =>
  new Promise((resolve) => {
    released.push(() => {
      const video = job.family.startsWith('wan22');
      const file = path.join(dir, `out-${job.id}.${video ? 'mp4' : 'png'}`);
      const bytes = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, ...Array.from({ length: 64 }, (_, i) => (i * 37 + job.id) % 256)]);
      fs.writeFileSync(file, bytes);
      const record = insertGeneration(db, job.params, job.family, file, null, job.workspaceId);
      written.set(record.id, bytes);
      resolve({ generationId: record.id });
    });
  });

const tick = () => new Promise((r) => setImmediate(r));

/** Lets every queued job finish. */
async function drain() {
  for (let i = 0; i < 100; i++) {
    await tick();
    const next = released.shift();
    if (next) next();
    else if (queue.list({}).every((j) => j.status !== 'queued' && j.status !== 'running')) return;
  }
}

before(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-api-ws-'));
  db = initDatabase(':memory:');
  queue = new JobQueue(db, fakeRunner, { cancelRunning: async () => undefined, currentWorkspaceId: () => currentWorkspace });
  const assemblies = new AssemblyManager(db, { ffmpeg: () => null, outputDir: () => path.join(dir, 'videos'), tempDir: () => path.join(dir, 'tmp') });
  service = new ApiService({
    db,
    queue,
    assemblies,
    ffmpeg: () => null,
    comfyAvailable: async () => true,
    imageSize: () => ({ width: 816, height: 1056 }),
    imagePreviewFallback: async () => null,
    workspaceId: () => currentWorkspace,
    setFavorite: () => '',
  });
  forge = createWorkspace(db, 'Color Book Forge');
  secret = createWorkspace(db, 'Secret Place');
  updateWorkspace(db, secret.id, { hidden: true });
});

after(() => fs.rmSync(dir, { recursive: true, force: true }));

const call = async (tool: string, args?: unknown) => (await service.callTool(tool, args)).data as any;
const rejects = (p: Promise<unknown>, code: string) => assert.rejects(p, (e: unknown) => e instanceof ApiError && e.code === code);
const workspaceOfJob = (id: number) => queue.get(id)?.workspaceId;

// -- workspace override ----------------------------------------------------------------------

test('every tool whose work or results depend on the workspace takes an optional "workspace"', () => {
  for (const name of ['generate_image', 'generate_video', 'list_jobs', 'list_library', 'list_styles', 'import_folder', 'assemble_video']) {
    const tool = TOOLS.find((t) => t.name === name);
    assert.ok(tool, name);
    assert.ok('workspace' in tool.inputSchema.properties, `${name} takes workspace`);
    assert.ok(!(tool.inputSchema.required ?? []).includes('workspace'), `${name}: workspace is optional`);
  }
});

test('generate_image with a workspace name queues the job there, ignoring case; without one it uses the current workspace', async () => {
  currentWorkspace = 1;
  const here = await call('generate_image', { prompt: 'in the current one', batch: 'ws' });
  assert.equal(workspaceOfJob(here.job_id), 1);
  const there = await call('generate_image', { prompt: 'in the forge', batch: 'ws', workspace: 'cOLOR bOOK fORGE' });
  assert.equal(workspaceOfJob(there.job_id), forge.id);
  await drain();
  const done = await call('get_job', { job_id: there.job_id });
  const row = db.prepare('SELECT workspace_id FROM generations WHERE id = ?').get(Number(done.item.id.replace('gen-', ''))) as { workspace_id: number };
  assert.equal(row.workspace_id, forge.id, 'the result is recorded in that workspace');
  assert.equal(
    (db.prepare('SELECT workspace_id FROM generations WHERE id = ?').get(Number((await call('get_job', { job_id: here.job_id })).item.id.replace('gen-', ''))) as { workspace_id: number }).workspace_id,
    1
  );
});

test('a job keeps the workspace it was queued in when the app moves to another one', async () => {
  currentWorkspace = 1;
  const first = await call('generate_image', { prompt: 'blocker' });
  const queued = await call('generate_image', { prompt: 'waits in the forge', workspace: 'Color Book Forge' });
  const plain = await call('generate_image', { prompt: 'waits in the current one' });
  currentWorkspace = secret.id; // the user switches while they wait
  assert.equal(workspaceOfJob(queued.job_id), forge.id);
  assert.equal(workspaceOfJob(plain.job_id), 1);
  await drain();
  for (const [job, expected] of [[queued, forge.id], [plain, 1], [first, 1]] as const) {
    const done = await call('get_job', { job_id: job.job_id });
    assert.equal(done.status, 'done');
    const row = db.prepare('SELECT workspace_id FROM generations WHERE id = ?').get(Number(done.item.id.replace('gen-', ''))) as { workspace_id: number };
    assert.equal(row.workspace_id, expected);
  }
  currentWorkspace = 1;
});

test('an unknown workspace is refused, lists the existing ones, and creates nothing', async () => {
  const before = listWorkspaces(db, true).length;
  await assert.rejects(
    service.callTool('generate_image', { prompt: 'x', workspace: 'Nope' }),
    (e: unknown) =>
      e instanceof ApiError &&
      e.code === 'not_found' &&
      e.message.startsWith('No workspace named "Nope". Existing workspaces: ') &&
      /"Color Book Forge"/.test(e.message) &&
      /"Secret Place"/.test(e.message) &&
      /"Default"/.test(e.message) &&
      /never through the API/.test(e.message)
  );
  assert.equal(listWorkspaces(db, true).length, before, 'no workspace was created');
  const jobsBefore = queue.list({}).length;
  for (const tool of ['generate_video', 'list_jobs', 'list_library', 'list_styles']) {
    await rejects(service.callTool(tool, { prompt: 'x', workspace: 'Nope' }), 'not_found');
  }
  await rejects(service.callTool('import_folder', { path: dir, workspace: 'Nope' }), 'not_found');
  await rejects(service.callTool('assemble_video', { clips: ['gen-1'], workspace: 'Nope' }), 'not_found');
  assert.equal(queue.list({}).length, jobsBefore, 'nothing was queued');
  // an empty name is a bad argument, not "use the current one"
  await rejects(service.callTool('generate_image', { prompt: 'x', workspace: '  ' }), 'invalid_argument');
  await rejects(service.callTool('generate_image', { prompt: 'x', workspace: 5 }), 'invalid_argument');
});

test('a hidden workspace can still be addressed by name; the app\'s own system workspace cannot', async () => {
  const job = await call('generate_image', { prompt: 'quiet', workspace: 'secret place' });
  assert.equal(workspaceOfJob(job.job_id), secret.id);
  await drain();
  styleExamplesWorkspaceId(db);
  await rejects(service.callTool('generate_image', { prompt: 'x', workspace: 'Style examples' }), 'not_found');
});

test('styles and elements are looked up in the named workspace', async () => {
  saveStyle(db, { name: 'Line art', text: 'clean black outlines' }, null, 1); // shared
  saveStyle(db, { name: 'Forge outfit', text: 'wearing an apron', kind: 'element' }, null, forge.id); // the forge's own
  currentWorkspace = 1;

  assert.deepEqual((await call('list_styles', {})).styles.map((s: any) => s.name), ['Line art']);
  assert.deepEqual((await call('list_styles', { workspace: 'Color Book Forge' })).styles.map((s: any) => s.name), ['Forge outfit', 'Line art']);
  assert.deepEqual((await call('list_styles', { workspace: 'Secret Place' })).styles.map((s: any) => s.name), ['Line art']);

  await assert.rejects(service.callTool('generate_image', { prompt: 'x', elements: ['Forge outfit'] }), (e: unknown) => e instanceof ApiError && e.code === 'not_found');
  const job = await call('generate_image', { prompt: 'a cat', style: 'line art', elements: ['forge outfit'], workspace: 'Color Book Forge' });
  assert.equal(job.prompt, 'a cat, wearing an apron, clean black outlines');
  await drain();
});

test('list_jobs and list_library narrow to the named workspace and are unchanged without one', async () => {
  currentWorkspace = 1;
  await call('generate_image', { prompt: 'one', batch: 'split' });
  const inForge = await call('generate_image', { prompt: 'two', batch: 'split', workspace: 'Color Book Forge' });
  await drain();

  const allJobs = (await call('list_jobs', { batch: 'split' })).jobs;
  assert.equal(allJobs.length, 2, 'no workspace: every workspace');
  const forgeJobs = (await call('list_jobs', { batch: 'split', workspace: 'color book forge' })).jobs;
  assert.deepEqual(forgeJobs.map((j: any) => j.job_id), [inForge.job_id]);
  assert.deepEqual((await call('list_jobs', { batch: 'split', workspace: 'Secret Place' })).jobs, []);

  assert.equal((await call('list_library', { batch: 'split' })).count, 2);
  const forgeItems = (await call('list_library', { batch: 'split', workspace: 'Color Book Forge' })).items;
  assert.equal(forgeItems.length, 1);
  assert.equal(forgeItems[0].prompt, 'two');
});

test('import_folder puts the files in the named workspace', async () => {
  const shoot = path.join(dir, 'shoot');
  fs.mkdirSync(shoot);
  fs.writeFileSync(path.join(shoot, 'a.png'), 'x');
  const imported = await call('import_folder', { path: shoot, batch: 'imp', workspace: 'Color Book Forge' });
  const row = db.prepare('SELECT workspace_id FROM imports WHERE id = ?').get(Number(imported.items[0].id.replace('imp-', ''))) as { workspace_id: number };
  assert.equal(row.workspace_id, forge.id);
  assert.equal((await call('list_library', { batch: 'imp', workspace: 'Color Book Forge' })).count, 1);
  assert.equal((await call('list_library', { batch: 'imp', workspace: 'Secret Place' })).count, 0);
});

test('list_capabilities names the workspaces and the current one', async () => {
  currentWorkspace = forge.id;
  const caps = await call('list_capabilities');
  assert.equal(caps.workspaces.current, 'Color Book Forge');
  assert.deepEqual(
    caps.workspaces.available.map((w: any) => [w.name, w.hidden]),
    [['Color Book Forge', false], ['Default', false], ['Secret Place', true]]
  );
  for (const family of caps.families) assert.ok(family.fields.workspace, `${family.family} documents workspace`);
  currentWorkspace = 1;
});

// -- exact sizes: z-image snaps to 16, everything else is unchanged ------------------------------

test('z-image keeps 816x1056 exactly (multiples of 16, min 256, max 2048); the old family name too', async () => {
  const job = await call('generate_image', { prompt: 'a page', width: 816, height: 1056 });
  assert.equal(job.family, 'z-image');
  assert.deepEqual([job.width, job.height], [816, 1056]);
  const old = await call('generate_image', { prompt: 'a page', family: 'z-image-turbo', width: 816, height: 1056 });
  assert.deepEqual([old.width, old.height], [816, 1056]);
  const odd = await call('generate_image', { prompt: 'a page', width: 1000, height: 700 });
  assert.deepEqual([odd.width, odd.height], [1008, 704], 'rounds to the nearest 16');
  const small = await call('generate_image', { prompt: 'a page', width: 100, height: 8000 });
  assert.deepEqual([small.width, small.height], [256, 2048], 'the min and max are unchanged');
  const defaults = await call('generate_image', { prompt: 'a page' });
  assert.deepEqual([defaults.width, defaults.height], [1024, 1024]);
  await drain();
});

test('the other image families still snap to 64, and video to 16 with its own limits', async () => {
  for (const family of ['qwen-image', 'flux-dev', 'hidream', 'ovis-image']) {
    const job = await call('generate_image', { prompt: 'a page', family, width: 816, height: 1056 });
    assert.deepEqual([job.width, job.height], [832, 1088], family);
  }
  const video = await call('generate_video', { prompt: 'x', width: 816, height: 1056 });
  assert.deepEqual([video.width, video.height], [816, 1056]);
  const wide = await call('generate_video', { prompt: 'x', width: 2000, height: 100 });
  assert.deepEqual([wide.width, wide.height], [1280, 256]);
  await drain();
});

test('image to image on z-image also uses the 16 grid; outpainting keeps its own sizing', async () => {
  const imported = await call('import_folder', { path: path.join(dir, 'shoot'), workspace: 'Color Book Forge' });
  const src = imported.items[0];
  const i2i = await call('generate_image', { prompt: 'x', source: src.id, width: 816, height: 1056 });
  assert.equal(i2i.family, 'z-image-i2i');
  assert.deepEqual([i2i.width, i2i.height], [816, 1056]);
  const out = await call('generate_image', { prompt: 'x', source: src.id, extend: { right: 256 } });
  assert.equal(out.family, 'z-image-outpaint');
  assert.equal(out.width % 64, 0, 'the outpainting canvas is still on the 64 grid');
  assert.equal(out.height % 64, 0);
  await drain();
});

test('list_capabilities reports multiple_of 16 for z-image and 64 for the other image families', async () => {
  const caps = await call('list_capabilities');
  const multiples = Object.fromEntries(caps.families.map((f: any) => [f.family, [f.fields.width.multiple_of, f.fields.height.multiple_of]]));
  assert.deepEqual(multiples['z-image'], [16, 16]);
  for (const family of ['qwen-image', 'flux-dev', 'hidream', 'ovis-image']) assert.deepEqual(multiples[family], [64, 64], family);
  assert.deepEqual(multiples['wan22-i2v'], [16, 16]);
  assert.deepEqual(multiples['wan22-t2v'], [16, 16]);
});

// -- get_image_file --------------------------------------------------------------------------------

async function makeGeneratedImage(workspace?: string) {
  const job = await call('generate_image', { prompt: 'bytes', ...(workspace ? { workspace } : {}) });
  await drain();
  return (await call('get_job', { job_id: job.job_id })).item as { id: string; path: string };
}

test('get_image_file returns the exact bytes of a generated image, base64, with its own mime type', async () => {
  const item = await makeGeneratedImage();
  const result = await service.callTool('get_image_file', { item_id: item.id });
  const data = result.data as any;
  assert.equal(data.item_id, item.id);
  assert.equal(data.mime_type, 'image/png');
  assert.equal(result.images?.length, 1);
  assert.equal(result.images![0].mimeType, 'image/png');
  const bytes = Buffer.from(result.images![0].data, 'base64');
  assert.ok(bytes.equals(fs.readFileSync(item.path)), 'byte for byte the file on disk');
  assert.ok(bytes.equals(written.get(Number(item.id.replace('gen-', '')))!));
  assert.equal(data.bytes, bytes.length);
});

test('get_image_file works for imported images (imp-N) and keeps a jpeg a jpeg', async () => {
  const folder = path.join(dir, 'jpegs');
  fs.mkdirSync(folder);
  const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4, 0xff, 0xd9]);
  fs.writeFileSync(path.join(folder, 'p.JPG'), jpeg);
  const item = (await call('import_folder', { path: folder })).items[0];
  assert.match(item.id, /^imp-\d+$/);
  const result = await service.callTool('get_image_file', { item_id: item.id });
  assert.equal(result.images![0].mimeType, 'image/jpeg');
  assert.ok(Buffer.from(result.images![0].data, 'base64').equals(jpeg));
});

test('get_image_file refuses a video or audio item and a missing file', async () => {
  const folder = path.join(dir, 'media');
  fs.mkdirSync(folder);
  fs.writeFileSync(path.join(folder, 'song.mp3'), 'x');
  fs.writeFileSync(path.join(folder, 'clip.mp4'), 'x');
  const items = (await call('import_folder', { path: folder })).items;
  for (const item of items) await rejects(service.callTool('get_image_file', { item_id: item.id }), 'invalid_argument');
  // a generated video
  const video = await call('generate_video', { prompt: 'x' });
  await drain();
  const generated = (await call('get_job', { job_id: video.job_id })).item;
  assert.equal(generated.kind, 'video');
  await rejects(service.callTool('get_image_file', { item_id: generated.id }), 'invalid_argument');
  // an image whose file is gone
  const gone = await makeGeneratedImage();
  fs.rmSync(gone.path);
  await rejects(service.callTool('get_image_file', { item_id: gone.id }), 'file_missing');
  await rejects(service.callTool('get_image_file', {}), 'invalid_argument');
});

test('get_image_file treats work done in the app exactly like an id that does not exist', async () => {
  const uiFile = path.join(dir, 'ui.png');
  fs.writeFileSync(uiFile, 'SECRET');
  const uiJob = queue.submit({ family: 'z-image', params: { prompt: 'SECRET prompt', width: 512, height: 512, seed: 1, steps: 4, cfg: 1 }, source: 'ui' });
  await drain();
  const uiGen = `gen-${(await queue.wait(uiJob.id)).generationId}`;
  const legacyFile = path.join(dir, 'legacy.png');
  fs.writeFileSync(legacyFile, 'SECRET');
  const legacyGen = `gen-${insertGeneration(db, { prompt: 'SECRET legacy', width: 512, height: 512, seed: 1, steps: 4, cfg: 1 }, 'z-image', legacyFile, null).id}`;

  const missing = await service.callTool('get_image_file', { item_id: 'gen-999999' }).catch((e) => e as ApiError);
  assert.ok(missing instanceof ApiError && missing.code === 'not_found');
  for (const id of [uiGen, legacyGen]) {
    const err = await service.callTool('get_image_file', { item_id: id }).catch((e) => e as ApiError);
    assert.ok(err instanceof ApiError && err.code === 'not_found', id);
    assert.equal(err.message.replace(id, 'X'), missing.message.replace('gen-999999', 'X'));
    assert.ok(!err.message.includes('SECRET'));
  }
  await rejects(service.callTool('get_image_file', { item_id: 'garbage' }), 'not_found');
  await rejects(service.callTool('get_image_file', { item_id: '../../etc/passwd' }), 'not_found');
});

test('get_image_file refuses files over the size cap with a clear error, and allows one right at it', async () => {
  const folder = path.join(dir, 'big');
  fs.mkdirSync(folder);
  const atCap = path.join(folder, 'at-cap.png');
  const overCap = path.join(folder, 'over-cap.png');
  for (const [file, size] of [[atCap, MAX_IMAGE_FILE_BYTES], [overCap, MAX_IMAGE_FILE_BYTES + 1]] as const) {
    fs.writeFileSync(file, '');
    fs.truncateSync(file, size);
  }
  const items = Object.fromEntries((await call('import_folder', { path: folder })).items.map((i: any) => [i.name, i]));
  await assert.rejects(
    service.callTool('get_image_file', { item_id: items['over-cap.png'].id }),
    (e: unknown) => e instanceof ApiError && e.code === 'too_large' && /25 MB/.test(e.message) && e.message.includes(items['over-cap.png'].id)
  );
  const ok = await service.callTool('get_image_file', { item_id: items['at-cap.png'].id });
  assert.equal((ok.data as any).bytes, MAX_IMAGE_FILE_BYTES);
  fs.rmSync(folder, { recursive: true, force: true });
});

// -- over HTTP ---------------------------------------------------------------------------------------

test('POST /v1/tools/get_image_file returns the bytes as JSON for a bearer-token caller', async () => {
  const item = await makeGeneratedImage('Color Book Forge');
  const token = 'test-token';
  const api: LocalApi = await startLocalApi({ token, preferredPort: 0, version: 'test', handler: (tool, args) => service.callTool(tool, args) });
  try {
    const url = `http://127.0.0.1:${api.port}/v1/tools/get_image_file`;
    const ok = await fetch(url, { method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ item_id: item.id }) });
    assert.equal(ok.status, 200);
    const body = (await ok.json()) as any;
    assert.equal(body.ok, true);
    assert.equal(body.data.item_id, item.id);
    assert.equal(body.images.length, 1);
    assert.equal(body.images[0].mimeType, 'image/png');
    assert.ok(Buffer.from(body.images[0].data, 'base64').equals(fs.readFileSync(item.path)));

    const unauth = await fetch(url, { method: 'POST', body: JSON.stringify({ item_id: item.id }) });
    assert.equal(unauth.status, 401);

    const unknown = await fetch(`http://127.0.0.1:${api.port}/v1/tools/generate_image`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: JSON.stringify({ prompt: 'x', workspace: 'Nope' }),
    });
    assert.equal(unknown.status, 404);
    const err = (await unknown.json()) as any;
    assert.equal(err.ok, false);
    assert.equal(err.error.code, 'not_found');
    assert.match(err.error.message, /Existing workspaces: .*"Color Book Forge"/);
  } finally {
    await api.close();
  }
});
