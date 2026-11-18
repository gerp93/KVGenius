import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { initDatabase, insertGeneration, getGenerationById } from './db';
import { JobQueue, JobRunner } from './jobQueue';
import { AssemblyManager } from './assembly';
import { ApiError, ApiService } from './apiService';
import { applyFavorite, FAVORITES_FOLDER } from './favorites';
import { createWorkspace } from './workspaces';
import { TOOLS } from '../shared/tools';

/** Favorites and pins through the API: set_item_flags, the favorite / pinned filters, and list_library's group_by. */
let dir: string;
let db: DatabaseSync;
let service: ApiService;
let queue: JobQueue;
let released: Array<() => void> = [];
const dirs = () => ({ images: path.join(dir, 'images'), videos: path.join(dir, 'videos'), gifs: path.join(dir, 'gifs'), legacy: path.join(dir, 'legacy') });

const fakeRunner: JobRunner = (job) =>
  new Promise((resolve) => {
    released.push(() => {
      const video = job.family.startsWith('wan22');
      fs.mkdirSync(video ? dirs().videos : dirs().images, { recursive: true });
      const file = path.join(video ? dirs().videos : dirs().images, `out-${job.id}.${video ? 'mp4' : 'png'}`);
      fs.writeFileSync(file, `bytes-${job.id}`);
      resolve({ generationId: insertGeneration(db, job.params, job.family, file, null, job.workspaceId).id });
    });
  });

const tick = () => new Promise((r) => setImmediate(r));
async function drain() {
  for (let i = 0; i < 100; i++) {
    await tick();
    const next = released.shift();
    if (next) next();
    else if (queue.list({}).every((j) => j.status !== 'queued' && j.status !== 'running')) return;
  }
}

before(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-api-flags-'));
  db = initDatabase(':memory:');
  queue = new JobQueue(db, fakeRunner, { cancelRunning: async () => undefined });
  const assemblies = new AssemblyManager(db, { ffmpeg: () => null, outputDir: () => path.join(dir, 'out'), tempDir: () => path.join(dir, 'tmp') });
  service = new ApiService({
    db,
    queue,
    assemblies,
    ffmpeg: () => null,
    comfyAvailable: async () => true,
    imageSize: () => null,
    imagePreviewFallback: async () => null,
    workspaceId: () => 1,
    // The same function the app wires in: the flag changes and the file moves into the favorites folder.
    setFavorite: (id, favorite) => applyFavorite(db, id, favorite, dirs(), ['wan22-i2v', 'wan22-t2v', 'wan22-v2v']).imagePath,
  });
  createWorkspace(db, 'Color Book Forge');
});

after(() => fs.rmSync(dir, { recursive: true, force: true }));

const call = async (tool: string, args?: unknown) => (await service.callTool(tool, args)).data as any;
const rejects = (p: Promise<unknown>, code: string) => assert.rejects(p, (e: unknown) => e instanceof ApiError && e.code === code);

async function make(prompt: string, extra: Record<string, unknown> = {}): Promise<string> {
  const job = await call('generate_image', { prompt, ...extra });
  await drain();
  return (await call('get_job', { job_id: job.job_id })).item.id as string;
}
const genId = (id: string) => Number(id.replace('gen-', ''));

test('set_item_flags is advertised, with item_ids required', () => {
  const tool = TOOLS.find((t) => t.name === 'set_item_flags');
  assert.ok(tool);
  assert.deepEqual(tool.inputSchema.required, ['item_ids']);
  for (const prop of ['favorite', 'pinned']) assert.ok(prop in tool.inputSchema.properties);
  const list = TOOLS.find((t) => t.name === 'list_library')!;
  for (const prop of ['favorite', 'pinned', 'group_by']) assert.ok(prop in list.inputSchema.properties, `list_library takes ${prop}`);
});

test('items show favorite and pinned (false at first), and set_item_flags changes them', async () => {
  const id = await make('a page of a cat', { batch: 'book' });
  const fresh = (await call('get_item', { item_id: id, preview: false })).item;
  assert.deepEqual([fresh.favorite, fresh.pinned], [false, false]);

  const result = await call('set_item_flags', { item_ids: [id], favorite: true, pinned: true });
  assert.equal(result.updated, 1);
  assert.deepEqual([result.items[0].favorite, result.items[0].pinned], [true, true]);
  assert.match(result.note, /favorites folder/);

  const again = (await call('get_item', { item_id: id, preview: false })).item;
  assert.deepEqual([again.favorite, again.pinned], [true, true]);
  assert.equal(getGenerationById(db, genId(id))?.favorite, true);
  assert.ok(getGenerationById(db, genId(id))?.pinned);

  // only the flag named changes
  const unpinned = await call('set_item_flags', { item_ids: [id], pinned: false });
  assert.deepEqual([unpinned.items[0].favorite, unpinned.items[0].pinned], [true, false]);
  assert.equal(unpinned.note, undefined);
  const unfavorited = await call('set_item_flags', { item_ids: [id], favorite: false });
  assert.deepEqual([unfavorited.items[0].favorite, unfavorited.items[0].pinned], [false, false]);
});

test('favoriting moves the file into the favorites folder, un-favoriting moves it back, and the item reports the new path', async () => {
  const id = await make('moving file');
  const before = (await call('get_item', { item_id: id, preview: false })).item.path as string;
  assert.ok(fs.existsSync(before));
  const favorited = (await call('set_item_flags', { item_ids: [id], favorite: true })).items[0];
  assert.notEqual(favorited.path, before);
  assert.equal(path.basename(path.dirname(favorited.path)), FAVORITES_FOLDER);
  assert.ok(fs.existsSync(favorited.path) && !fs.existsSync(before));
  // get_image_file follows the move
  const file = await service.callTool('get_image_file', { item_id: id });
  assert.ok(Buffer.from(file.images![0].data, 'base64').length > 0);
  const back = (await call('set_item_flags', { item_ids: [id], favorite: false })).items[0];
  assert.equal(back.path, before);
  assert.ok(fs.existsSync(before));
});

test('set_item_flags validates everything first: a bad id, an import, an app item or no flag changes nothing', async () => {
  const good = await make('stays unflagged');
  const folder = path.join(dir, 'imp');
  fs.mkdirSync(folder);
  fs.writeFileSync(path.join(folder, 'a.png'), 'x');
  const imp = (await call('import_folder', { path: folder })).items[0].id;

  await rejects(service.callTool('set_item_flags', { item_ids: [good, 'gen-999999'], favorite: true }), 'not_found');
  await rejects(service.callTool('set_item_flags', { item_ids: [good, imp], favorite: true }), 'invalid_argument');
  await rejects(service.callTool('set_item_flags', { item_ids: [good] }), 'invalid_argument');
  await rejects(service.callTool('set_item_flags', { item_ids: [], favorite: true }), 'invalid_argument');
  await rejects(service.callTool('set_item_flags', { favorite: true }), 'invalid_argument');
  await rejects(service.callTool('set_item_flags', { item_ids: [good], favorite: 'yes' }), 'invalid_argument');
  assert.equal(getGenerationById(db, genId(good))?.favorite, false, 'the good id was not touched by the failed calls');

  // Work done in the app looks nonexistent, like everywhere else.
  const uiJob = queue.submit({ family: 'z-image', params: { prompt: 'SECRET', width: 512, height: 512, seed: 1, steps: 4, cfg: 1 }, source: 'ui' });
  await drain();
  const uiGen = `gen-${(await queue.wait(uiJob.id)).generationId}`;
  const missing = await service.callTool('set_item_flags', { item_ids: ['gen-999998'], pinned: true }).catch((e) => e as ApiError);
  const err = await service.callTool('set_item_flags', { item_ids: [uiGen], pinned: true }).catch((e) => e as ApiError);
  assert.ok(err instanceof ApiError && err.code === 'not_found');
  assert.equal(err.message.replace(uiGen, 'X'), (missing as ApiError).message.replace('gen-999998', 'X'));
  assert.equal(getGenerationById(db, genId(uiGen))?.pinned, false, 'the app item was not pinned');
});

test('set_item_flags flags several items at once (duplicates counted once)', async () => {
  const a = await make('multi one');
  const b = await make('multi two');
  const result = await call('set_item_flags', { item_ids: [a, b, a], favorite: true });
  assert.equal(result.updated, 2);
  assert.ok(result.items.every((i: any) => i.favorite));
});

test('list_library filters by favorite and pinned, and flag filters leave imports out', async () => {
  const keep = await make('filter keep', { batch: 'filter' });
  const pin = await make('filter pin', { batch: 'filter' });
  const plain = await make('filter plain', { batch: 'filter' });
  const folder = path.join(dir, 'impf');
  fs.mkdirSync(folder);
  fs.writeFileSync(path.join(folder, 'b.png'), 'x');
  await call('import_folder', { path: folder, batch: 'filter' });
  await call('set_item_flags', { item_ids: [keep], favorite: true });
  await call('set_item_flags', { item_ids: [pin], pinned: true });

  const ids = async (extra: Record<string, unknown>) => (await call('list_library', { batch: 'filter', ...extra })).items.map((i: any) => i.id).sort();
  assert.deepEqual(await ids({ favorite: true }), [keep]);
  assert.deepEqual(await ids({ pinned: true }), [pin]);
  assert.deepEqual(await ids({ favorite: false, pinned: false }), [plain]);
  assert.deepEqual(await ids({ favorite: false }), [pin, plain].sort());
  assert.equal((await ids({})).length, 4, 'with no flag filter the import is listed');
  await rejects(service.callTool('list_library', { favorite: 'yes' }), 'invalid_argument');
});

test('list_library group_by prompt collects candidates per prompt, with counts and a cover (pinned > favorite > newest)', async () => {
  const page1 = [await make('page one: a fox', { batch: 'grp' }), await make('page one: a fox', { batch: 'grp' }), await make('page one: a fox', { batch: 'grp' })];
  const page2 = [await make('page two: an owl', { batch: 'grp' }), await make('page two: an owl', { batch: 'grp' })];
  const single = await make('page three: a bear', { batch: 'grp' });
  await call('set_item_flags', { item_ids: [page1[0]], favorite: true });
  await call('set_item_flags', { item_ids: [page1[1]], pinned: true });

  const result = await call('list_library', { batch: 'grp', group_by: 'prompt' });
  assert.equal(result.group_by, 'prompt');
  assert.equal(result.truncated, false);
  assert.deepEqual(result.groups.map((g: any) => g.prompt), ['page three: a bear', 'page two: an owl', 'page one: a fox'], 'newest group first');
  const fox = result.groups[2];
  assert.equal(fox.count, 3);
  assert.equal(fox.favorites, 1);
  assert.equal(fox.pinned, 1);
  assert.equal(fox.cover.id, page1[1], 'the pinned item is the cover');
  assert.deepEqual(fox.item_ids, [...page1].reverse(), 'newest first');
  const owl = result.groups[1];
  assert.equal(owl.cover.id, page2[1], 'with no flags the newest is the cover');
  assert.equal(result.groups[0].cover.id, single);

  // the filter applies to items first: only favorites -> one group of one
  const favs = await call('list_library', { batch: 'grp', group_by: 'prompt', favorite: true });
  assert.deepEqual(favs.groups.map((g: any) => [g.prompt, g.count]), [['page one: a fox', 1]]);
  // limit counts groups
  assert.equal((await call('list_library', { batch: 'grp', group_by: 'prompt', limit: 2 })).groups.length, 2);
});

test('group_by batch groups by label and respects the workspace', async () => {
  const a = await make('b1', { batch: 'alpha' });
  await make('b2', { batch: 'alpha' });
  await make('b3', { batch: 'beta', workspace: 'Color Book Forge' });
  await call('set_item_flags', { item_ids: [a], favorite: true });

  const all = await call('list_library', { group_by: 'batch', workspace: 'Color Book Forge' });
  assert.deepEqual(all.groups.map((g: any) => [g.batch, g.count]), [['beta', 1]]);
  const alpha = (await call('list_library', { group_by: 'batch', batch: 'alpha' })).groups[0];
  assert.deepEqual([alpha.batch, alpha.count, alpha.favorites, alpha.cover.id], ['alpha', 2, 1, a]);
  await rejects(service.callTool('list_library', { group_by: 'color' }), 'invalid_argument');
});

test('flags are only on generated items; an out-of-scope item never appears in a group', async () => {
  const file = path.join(dir, 'legacy.png');
  fs.writeFileSync(file, 'x');
  insertGeneration(db, { prompt: 'SECRET legacy', width: 512, height: 512, seed: 1, steps: 4, cfg: 1 }, 'z-image', file, null);
  const grouped = await call('list_library', { group_by: 'prompt' });
  assert.ok(!JSON.stringify(grouped).includes('SECRET'));
});
