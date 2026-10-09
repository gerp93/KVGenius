import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { initDatabase, insertGeneration, getGenerationById } from './db';
import { saveStyle } from './styles';
import { saveModelProfile } from './modelProfiles';
import { JobQueue, JobRunner } from './jobQueue';
import { AssemblyManager } from './assembly';
import { ApiError, ApiService } from './apiService';
import { findFfmpeg } from './mediaTools';

const ff = findFfmpeg();
let dir: string;
let db: DatabaseSync;
let service: ApiService;
let queue: JobQueue;
let released: Array<() => void> = [];
let runnerCalls: string[] = [];

/** The runner records a real generation row (like the app's does) once the test lets it finish. */
const fakeRunner: JobRunner = (job) =>
  new Promise((resolve) => {
    runnerCalls.push(job.family);
    released.push(() => {
      const file = path.join(dir, `out-${job.id}.${job.family.startsWith('wan22') ? 'mp4' : 'png'}`);
      fs.writeFileSync(file, 'x');
      resolve({ generationId: insertGeneration(db, job.params, job.family, file, null).id });
    });
  });

const tick = () => new Promise((r) => setImmediate(r));

before(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-api-'));
  db = initDatabase(':memory:');
  queue = new JobQueue(db, fakeRunner, { cancelRunning: async () => undefined });
  const assemblies = new AssemblyManager(db, { ffmpeg: () => ff, outputDir: () => path.join(dir, 'videos'), tempDir: () => path.join(dir, 'tmp') });
  service = new ApiService({
    db,
    queue,
    assemblies,
    ffmpeg: () => ff,
    comfyAvailable: async () => true,
    imageSize: () => ({ width: 1600, height: 900 }),
    imagePreviewFallback: async () => null,
  });
});

after(() => fs.rmSync(dir, { recursive: true, force: true }));

const call = async (tool: string, args?: unknown) => (await service.callTool(tool, args)).data as any;
const rejects = (p: Promise<unknown>, code: string) =>
  assert.rejects(p, (e: unknown) => e instanceof ApiError && e.code === code);

test('unknown tools and non-object arguments are rejected', async () => {
  await rejects(service.callTool('nope', {}), 'unknown_tool');
  await rejects(service.callTool('list_jobs', [1]), 'invalid_argument');
});

test('list_capabilities describes both families', async () => {
  const caps = await call('list_capabilities');
  assert.equal(caps.comfyui_reachable, true);
  assert.deepEqual(caps.families.map((f: any) => f.family), ['z-image', 'wan22-i2v', 'wan22-t2v']);
});

test('import_folder imports media in natural filename order, ignores the rest, and labels the batch', async () => {
  const shoot = path.join(dir, 'shoot');
  fs.mkdirSync(shoot);
  for (const n of ['img10.png', 'img2.png', 'img1.jpg', 'notes.txt']) fs.writeFileSync(path.join(shoot, n), 'x');
  fs.writeFileSync(path.join(shoot, 'song.mp3'), 'x');

  const result = await call('import_folder', { path: shoot, batch: 'mv' });
  assert.deepEqual(result.items.map((i: any) => i.name), ['img1.jpg', 'img2.png', 'img10.png', 'song.mp3']);
  assert.deepEqual(result.items.map((i: any) => i.kind), ['image', 'image', 'image', 'audio']);
  assert.equal(result.items[0].batch, 'mv');
  assert.equal(result.items[0].width, 1600);

  const again = await call('import_folder', { path: shoot });
  assert.equal(again.items[0].id, result.items[0].id, 're-importing returns the same ids');
  assert.equal(again.items[0].batch, 'mv', 'and keeps the batch when none is given');
});

test('import_folder validates its path', async () => {
  await rejects(service.callTool('import_folder', { path: 'relative/dir' }), 'invalid_argument');
  await rejects(service.callTool('import_folder', { path: path.join(dir, 'missing') }), 'not_found');
  const empty = path.join(dir, 'empty');
  fs.mkdirSync(empty);
  await rejects(service.callTool('import_folder', { path: empty }), 'no_media');
  fs.writeFileSync(path.join(dir, 'a.txt'), 'x');
  await rejects(service.callTool('import_folder', { path: path.join(dir, 'a.txt') }), 'invalid_argument');
});

test('generate_image snaps sizes, applies defaults, and queues a job', async () => {
  const job = await call('generate_image', { prompt: 'a red fox', width: 1000, height: 700, batch: 'b1' });
  assert.equal(job.status === 'queued' || job.status === 'running', true);
  assert.equal(job.width, 1024);
  assert.equal(job.height, 704);
  assert.equal(job.kind, 'image');
  assert.equal(job.batch, 'b1');
  released.shift()?.();
  const done = await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
  assert.equal(done.status, 'done');
  assert.equal(done.item.kind, 'image');
  assert.equal(done.item.batch, 'b1');
  assert.match(done.item.id, /^gen-\d+$/);
});

test('generate_image still accepts the old family key, and queues the job under the current one', async () => {
  const job = await call('generate_image', { prompt: 'a red fox', family: 'z-image-turbo' });
  assert.equal(job.family, 'z-image');
  released.shift()?.();
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
});

function lastJobParams(): Record<string, unknown> {
  const row = db.prepare('SELECT params FROM jobs ORDER BY id DESC LIMIT 1').get() as { params: string };
  return JSON.parse(row.params) as Record<string, unknown>;
}

test('generate_image with a saved model sends its files and sampler, and defaults steps and cfg to its own', async () => {
  saveModelProfile(db, {
    family: 'z-image',
    name: 'Photoreal',
    files: { diffusionModel: 'photoreal.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' },
    sampler: { steps: 30, cfg: 4, sampler: 'euler', scheduler: 'karras', shift: 3 },
  });
  const job = await call('generate_image', { prompt: 'a red fox', model: 'photoreal' });
  const params = lastJobParams();
  assert.equal(params.steps, 30);
  assert.equal(params.cfg, 4);
  assert.equal(params.modelName, 'Photoreal');
  assert.deepEqual(params.modelSettings, {
    files: { diffusionModel: 'photoreal.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' },
    sampler: 'euler',
    scheduler: 'karras',
    shift: 3,
  });
  released.shift()?.();
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });

  // steps may be overridden, up to the wider limit a model allows
  const second = await call('generate_image', { prompt: 'a red fox', model: 'Photoreal', steps: 60 });
  assert.equal(lastJobParams().steps, 60);
  released.shift()?.();
  await call('get_job', { job_id: second.job_id, wait_seconds: 5 });
});

test('generate_image without a model, or with the built-in one, sends the template as shipped', async () => {
  const plain = await call('generate_image', { prompt: 'a red fox' });
  assert.equal(lastJobParams().modelSettings, undefined);
  released.shift()?.();
  await call('get_job', { job_id: plain.job_id, wait_seconds: 5 });
  const builtIn = await call('generate_image', { prompt: 'a red fox', model: 'Z Image Turbo' });
  assert.equal(lastJobParams().modelSettings, undefined);
  released.shift()?.();
  await call('get_job', { job_id: builtIn.job_id, wait_seconds: 5 });
  // the shipped limits still apply to it
  await assert.rejects(() => call('generate_image', { prompt: 'a red fox', steps: 50 }), /steps/);
});

test('an unknown model is refused with the saved names, and list_models shows what there is', async () => {
  await assert.rejects(() => call('generate_image', { prompt: 'a red fox', model: 'Nope' }), /No model named "Nope".*"Photoreal"/);
  const listed = await call('list_models', {});
  assert.deepEqual(listed.models.map((m: { name: string }) => m.name), ['Z Image Turbo', 'Wan 2.2 image to video', 'Wan 2.2 text to video', 'Photoreal']);
  assert.equal(listed.models[0].built_in, true);
  assert.equal(listed.models[1].tool, 'generate_video');
});

test('generate_image with a library picture as source is image to image: its shape, the strength and the picture are sent', async () => {
  const src = (await call('list_library', { origin: 'imported', kind: 'image' })).items.find((i: any) => i.name === 'img1.jpg');
  const job = await call('generate_image', { prompt: 'a fox', source: src.id, strength: 0.4, model: 'Photoreal' });
  assert.equal(job.family, 'z-image-i2i');
  assert.equal(job.width, 1024);
  assert.equal(job.height, 576); // img1 is 1600x900 -> long side 1024, shape kept, snapped to 64
  const params = lastJobParams();
  assert.equal(params.denoise, 0.4);
  assert.equal(params.sourceImagePath, src.path);
  assert.equal(params.modelName, 'Photoreal', 'a Z-Image model applies to image to image too');
  released.shift()?.();
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
  // no strength: the default; an explicit size wins over the picture's shape
  const second = await call('generate_image', { prompt: 'a fox', source: src.id, width: 512, height: 512 });
  assert.equal(lastJobParams().denoise, 0.6);
  assert.equal(second.width, 512);
  released.shift()?.();
  await call('get_job', { job_id: second.job_id, wait_seconds: 5 });
});

test('generate_image with source and extend is outpainting: the whole extended canvas is the size, the padding and picture are sent', async () => {
  const src = (await call('list_library', { origin: 'imported', kind: 'image' })).items.find((i: any) => i.name === 'img1.jpg');
  const job = await call('generate_image', { prompt: 'a fox', source: src.id, extend: { left: 400, right: 400 }, width: 64, strength: 0.2 });
  assert.equal(job.family, 'z-image-outpaint');
  // 1600x900 + 800 wide = 2400x900 -> long side 1536 -> 1536 x 576
  assert.equal(job.width, 1536);
  assert.equal(job.height, 576);
  const params = lastJobParams();
  assert.deepEqual(params.outpaint, { left: 400, top: 0, right: 400, bottom: 0 });
  assert.equal(params.sourceImagePath, src.path);
  assert.equal(params.denoise, 0.2, 'strength is how much of the new area is re-drawn');
  released.shift()?.();
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
  // without a strength the outpainting default is used
  const second = await call('generate_image', { prompt: 'a fox', source: src.id, extend: { right: 128 } });
  assert.equal(lastJobParams().denoise, 0.8);
  released.shift()?.();
  await call('get_job', { job_id: second.job_id, wait_seconds: 5 });
});

test('extend needs a source and a side above 0; the outpainting family cannot be named', async () => {
  const src = (await call('list_library', { origin: 'imported', kind: 'image' })).items.find((i: any) => i.name === 'img1.jpg');
  await assert.rejects(() => call('generate_image', { prompt: 'x', extend: { left: 64 } }), /needs a "source"/);
  await assert.rejects(() => call('generate_image', { prompt: 'x', source: src.id, extend: { left: 0 } }), /above 0/);
  await assert.rejects(() => call('generate_image', { prompt: 'x', family: 'z-image-outpaint' }), /Outpainting is not a family/);
});

test('image to image is asked for with a source, not by naming its family; the source must be a picture', async () => {
  await assert.rejects(() => call('generate_image', { prompt: 'x', family: 'z-image-i2i' }), /pass `source`/);
  await assert.rejects(() => call('generate_image', { prompt: 'x', family: 'z-image-inpaint' }), /Inpainting is not available/);
  await assert.rejects(() => call('generate_image', { prompt: 'x', source: 'imp-9999' }), /no library item/);
  const song = (await call('list_library', { kind: 'audio' })).items[0];
  await assert.rejects(() => call('generate_image', { prompt: 'x', source: song.id }), /image/i);
});

test('generate_image without a style sends the prompt untouched, with one it appends the style', async () => {
  const plain = await call('generate_image', { prompt: 'a red fox' });
  assert.equal(plain.prompt, 'a red fox');
  assert.equal(plain.style, null);
  released.shift()?.();
  await call('get_job', { job_id: plain.job_id, wait_seconds: 5 });

  saveStyle(db, { name: '1930s movie poster', text: 'bold lithograph, limited palette' });
  const styled = await call('generate_image', { prompt: 'a red fox', style: '1930S MOVIE POSTER' });
  assert.equal(styled.prompt, 'a red fox, bold lithograph, limited palette');
  assert.equal(styled.style, '1930s movie poster');
  released.shift()?.();
  const done = await call('get_job', { job_id: styled.job_id, wait_seconds: 5 });
  const record = getGenerationById(db, Number(done.item.id.replace('gen-', '')));
  assert.equal(record?.prompt, 'a red fox, bold lithograph, limited palette');
  assert.equal(record?.styleName, '1930s movie poster');
});

test('generate_image names the saved styles when the one asked for does not exist', async () => {
  await assert.rejects(
    service.callTool('generate_image', { prompt: 'x', style: 'nope' }),
    (e: unknown) => e instanceof ApiError && e.code === 'not_found' && /"1930s movie poster"/.test(e.message)
  );
});

test('list_styles returns the saved styles', async () => {
  const result = await call('list_styles');
  assert.deepEqual(result.styles, [{ name: '1930s movie poster', text: 'bold lithograph, limited palette', kind: 'style' }]);
});

test('generate_image rejects bad input', async () => {
  await rejects(service.callTool('generate_image', {}), 'invalid_argument');
  await rejects(service.callTool('generate_image', { prompt: 'x', steps: 99 }), 'invalid_argument');
  await rejects(service.callTool('generate_image', { prompt: 'x', family: 'wan22-i2v' }), 'invalid_argument');
  await rejects(service.callTool('generate_image', { prompt: 'x', width: 'big' }), 'invalid_argument');
});

test('generate_video takes the source image path and keeps its aspect ratio', async () => {
  const src = (await call('list_library', { origin: 'imported', kind: 'image' })).items.find((i: any) => i.name === 'img1.jpg');
  const job = await call('generate_video', { prompt: 'slow pan', source: src.id, seconds: 5, batch: 'mv' });
  assert.equal(job.kind, 'video');
  assert.equal(job.width, 640);
  assert.equal(job.height, 368); // 640 * 900/1600 = 360 -> 22.5 sixteens, rounds up to 368 (same rule as the Generate page)
  assert.equal(job.seconds, 5);
  assert.equal(job.source_image, src.path);
  released.shift()?.();
});

test('generate_video without a source is text to video (640 square by default); the two kinds cannot be mixed up', async () => {
  const job = await call('generate_video', { prompt: 'a fox running through snow', seconds: 5 });
  assert.equal(job.kind, 'video');
  assert.equal(job.family, 'wan22-t2v');
  assert.equal(job.width, 640);
  assert.equal(job.height, 640);
  assert.equal(job.source_image ?? null, null);
  released.shift()?.();
  const src = (await call('list_library', { origin: 'imported', kind: 'image' })).items.find((i: any) => i.name === 'img1.jpg');
  await rejects(service.callTool('generate_video', { prompt: 'x', source: src.id, family: 'wan22-t2v' }), 'invalid_argument');
  await rejects(service.callTool('generate_video', { prompt: 'x', family: 'wan22-i2v' }), 'invalid_argument');
});

test('generate_video with a saved video model sends its files; a model of the other family is refused', async () => {
  const files = { highNoiseModel: 'hi.safetensors', lowNoiseModel: 'lo.safetensors', textEncoder: 'umt5.safetensors', vae: 'v.safetensors', highNoiseLora: 'lh.safetensors', lowNoiseLora: 'll.safetensors' };
  saveModelProfile(db, { family: 'wan22-i2v', name: 'Wan custom', files, sampler: { steps: 1, cfg: 0, sampler: '', scheduler: '', shift: 0 } });
  const src = (await call('list_library', { origin: 'imported', kind: 'image' })).items.find((i: any) => i.name === 'img1.jpg');
  const job = await call('generate_video', { prompt: 'slow pan', source: src.id, model: 'wan custom' });
  const params = lastJobParams();
  assert.equal(params.modelName, 'Wan custom');
  assert.deepEqual(params.modelSettings, { files });
  assert.equal(params.steps, 8, 'the video template ignores steps; the default is unchanged');
  released.shift()?.();
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
  // an image model cannot be used for video, nor a video model for an image
  await assert.rejects(() => call('generate_video', { prompt: 'x', source: src.id, model: 'Photoreal' }), /"z-image" family, not "wan22-i2v"/);
  await assert.rejects(() => call('generate_image', { prompt: 'x', model: 'Wan custom' }), /"wan22-i2v" family, not "z-image"/);
});
test('generate_video validates the source item', async () => {
  await rejects(service.callTool('generate_video', { prompt: 'x', source: 'imp-9999' }), 'not_found');
  await rejects(service.callTool('generate_video', { prompt: 'x', source: 'garbage' }), 'not_found');
  const song = (await call('list_library', { kind: 'audio' })).items[0];
  await rejects(service.callTool('generate_video', { prompt: 'x', source: song.id }), 'invalid_argument');
});

test('jobs run one at a time; list_jobs filters and counts; cancel works on waiting jobs and batches', async () => {
  for (let i = 0; i < 50 && (await call('list_jobs', { status: 'running' })).jobs.length > 0; i++) await tick();
  released = [];
  runnerCalls = [];
  const a = await call('generate_image', { prompt: 'a', batch: 'q' });
  const b = await call('generate_image', { prompt: 'b', batch: 'q' });
  const c = await call('generate_image', { prompt: 'c', batch: 'other' });
  await tick();
  assert.equal(runnerCalls.length, 1);

  const listed = await call('list_jobs', { batch: 'q' });
  assert.deepEqual(listed.jobs.map((j: any) => j.job_id), [b.job_id, a.job_id]);
  assert.equal(listed.counts.queued, 1);
  assert.equal(listed.counts.running, 1);

  const cancelledOne = await call('cancel_job', { job_id: b.job_id });
  assert.equal(cancelledOne.cancelled, true);
  assert.equal(cancelledOne.job.status, 'cancelled');

  const batchCancel = await call('cancel_job', { batch: 'other' });
  assert.equal(batchCancel.cancelled_waiting_jobs, 1);
  assert.equal((await call('get_job', { job_id: c.job_id })).status, 'cancelled');

  released.shift()?.();
  assert.equal((await call('get_job', { job_id: a.job_id, wait_seconds: 5 })).status, 'done');
  assert.equal(runnerCalls.length, 1, 'cancelled jobs never ran');
});

test('cancel_job wants exactly one target; get_job needs a real job', async () => {
  await rejects(service.callTool('cancel_job', {}), 'invalid_argument');
  await rejects(service.callTool('cancel_job', { job_id: 1, batch: 'x' }), 'invalid_argument');
  await rejects(service.callTool('cancel_job', { job_id: 99999 }), 'not_found');
  await rejects(service.callTool('get_job', { job_id: 99999 }), 'not_found');
  await rejects(service.callTool('get_job', {}), 'invalid_argument');
});

test('list_library filters by batch, kind and origin', async () => {
  const mv = await call('list_library', { batch: 'mv' });
  assert.ok(mv.items.length >= 4, 'imports and the generated clip share the batch');
  assert.ok((await call('list_library', { kind: 'video', batch: 'mv' })).items.every((i: any) => i.kind === 'video'));
  assert.ok((await call('list_library', { origin: 'generated' })).items.every((i: any) => i.origin === 'generated'));
});

test('get_item reports the file and rejects unknown ids', async () => {
  const song = (await call('list_library', { kind: 'audio' })).items[0];
  const got = await service.callTool('get_item', { item_id: song.id });
  assert.equal((got.data as any).item.id, song.id);
  assert.equal(got.images, undefined, 'audio has no preview');
  await rejects(service.callTool('get_item', { item_id: 'gen-99999' }), 'not_found');
});

test('work done in the app is invisible and unusable to clients', async () => {
  for (let i = 0; i < 50 && (await call('list_jobs', { status: 'running' })).jobs.length > 0; i++) await tick();
  released = [];
  const uiParams = { prompt: 'SECRET prompt typed in the app', width: 512, height: 512, seed: 1, steps: 4, cfg: 1 };

  // A generation made in the app, through the same queue.
  const uiJob = queue.submit({ family: 'z-image', params: uiParams, source: 'ui', batch: 'shared' });
  released.shift()?.();
  const uiDone = await queue.wait(uiJob.id);
  assert.equal(uiDone.status, 'done');
  const uiGen = `gen-${uiDone.generationId}`;

  // One from before jobs were recorded at all (no job row).
  const legacyFile = path.join(dir, 'legacy.png');
  fs.writeFileSync(legacyFile, 'x');
  const legacyGen = `gen-${insertGeneration(db, { ...uiParams, prompt: 'SECRET legacy prompt' }, 'z-image', legacyFile, null).id}`;

  // Not in any listing, by any filter.
  for (const filter of [{}, { kind: 'image' }, { origin: 'generated' }, { batch: 'shared' }]) {
    const items = (await call('list_library', { ...filter, limit: 200 })).items;
    assert.ok(!items.some((i: any) => [uiGen, legacyGen].includes(i.id)), `listed with ${JSON.stringify(filter)}`);
    assert.ok(!JSON.stringify(items).includes('SECRET'), 'no prompt text leaks');
  }

  // Indistinguishable from an id that does not exist.
  const missing = await service.callTool('get_item', { item_id: 'gen-999999' }).catch((e) => e as ApiError);
  for (const id of [uiGen, legacyGen]) {
    const err = await service.callTool('get_item', { item_id: id }).catch((e) => e as ApiError);
    assert.ok(err instanceof ApiError && err.code === 'not_found');
    assert.equal(err.message.replace(id, 'X'), (missing as ApiError).message.replace('gen-999999', 'X'));
    await rejects(service.callTool('probe_media', { item_id: id }), 'not_found');
    await rejects(service.callTool('generate_video', { prompt: 'x', source: id }), 'not_found');
    await rejects(service.callTool('assemble_video', { clips: [id] }), 'not_found');
  }

  // Its job is invisible too: listing, lookup and cancel.
  assert.ok(!(await call('list_jobs', {})).jobs.some((j: any) => j.job_id === uiJob.id));
  assert.ok(!JSON.stringify(await call('list_jobs', { batch: 'shared' })).includes('SECRET'));
  await rejects(service.callTool('get_job', { job_id: uiJob.id }), 'not_found');
  await rejects(service.callTool('cancel_job', { job_id: uiJob.id }), 'not_found');

  // A client cancelling a batch label cannot reach a waiting app job that happens to share it.
  const clientJob = await call('generate_image', { prompt: 'client work', batch: 'shared' });
  const waitingUi = queue.submit({ family: 'z-image', params: uiParams, source: 'ui', batch: 'shared' });
  await tick();
  assert.equal(queue.get(waitingUi.id)?.status, 'queued');
  assert.equal((await call('cancel_job', { batch: 'shared' })).cancelled_waiting_jobs, 0);
  assert.equal(queue.get(waitingUi.id)?.status, 'queued', 'the app job is untouched');
  released.shift()?.();
  assert.equal((await call('get_job', { job_id: clientJob.job_id, wait_seconds: 5 })).status, 'done');
  await tick();
  released.shift()?.();
  assert.equal((await queue.wait(waitingUi.id)).status, 'done');

  // What the client made itself is still fully visible, and usable.
  const mine = (await call('get_job', { job_id: clientJob.job_id })).item;
  assert.equal(mine.prompt, 'client work');
  assert.equal((await call('get_item', { item_id: mine.id })).item.id, mine.id);
  assert.ok((await call('list_library', { batch: 'shared' })).items.some((i: any) => i.id === mine.id));
});

// -- with a real ffmpeg -----------------------------------------------------------------------

const withFfmpeg = ff ? test : test.skip;

function makeMedia() {
  const media = path.join(dir, 'media');
  fs.mkdirSync(media, { recursive: true });
  const run = (...args: string[]) => execFileSync(ff!.ffmpeg, ['-y', '-v', 'error', ...args], { stdio: 'pipe' });
  for (const n of [1, 2, 3]) {
    run('-f', 'lavfi', '-i', `testsrc=size=128x96:rate=16:duration=2`, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path.join(media, `clip${n}.mp4`));
  }
  run('-f', 'lavfi', '-i', 'sine=frequency=440:duration=10', path.join(media, 'song.wav'));
  run('-f', 'lavfi', '-i', 'testsrc=size=320x240', '-frames:v', '1', path.join(media, 'still.png'));
  return media;
}

withFfmpeg('probe_media and get_item previews work on real files', async () => {
  const media = makeMedia();
  const imported = await call('import_folder', { path: media, batch: 'real' });
  const byName = Object.fromEntries(imported.items.map((i: any) => [i.name, i]));
  assert.equal(byName['clip1.mp4'].kind, 'video');
  assert.ok(Math.abs(byName['clip1.mp4'].seconds - 2) < 0.1);
  assert.equal(byName['clip1.mp4'].width, 128);

  const probe = await call('probe_media', { item_id: byName['song.wav'].id });
  assert.ok(Math.abs(probe.duration_seconds - 10) < 0.1);
  assert.equal(probe.has_audio, true);
  assert.equal(probe.has_video, false);

  const videoProbe = await call('probe_media', { item_id: byName['clip1.mp4'].id });
  assert.equal(videoProbe.fps, 16);

  for (const name of ['clip1.mp4', 'still.png']) {
    const item = await service.callTool('get_item', { item_id: byName[name].id });
    assert.equal(item.images?.length, 1, `${name} has a preview`);
    const jpeg = Buffer.from(item.images![0].data, 'base64');
    assert.deepEqual([...jpeg.subarray(0, 2)], [0xff, 0xd8], 'a JPEG');
  }
  const noPreview = await service.callTool('get_item', { item_id: byName['clip1.mp4'].id, preview: false });
  assert.equal(noPreview.images, undefined);
});

async function assemble(args: Record<string, unknown>) {
  const started = await call('assemble_video', { ...args, wait_seconds: 60 });
  assert.equal(started.status, 'done', `assembly failed: ${started.error}`);
  return started;
}

withFfmpeg('assemble_video: hard cuts (stream copy) with the track under them', async () => {
  const items = (await call('list_library', { origin: 'imported', batch: 'real' })).items;
  const clips = items.filter((i: any) => i.kind === 'video').sort((a: any, b: any) => a.name.localeCompare(b.name)).map((i: any) => i.id);
  const song = items.find((i: any) => i.kind === 'audio').id;
  assert.equal(clips.length, 3);

  const result = await assemble({ clips, audio: song, name: 'cuts', batch: 'real' });
  assert.equal(result.item.origin, 'assembled');
  assert.equal(result.item.name, 'cuts.mp4');
  assert.ok(fs.existsSync(result.item.path));
  const probe = await call('probe_media', { item_id: result.item.id });
  assert.ok(Math.abs(probe.duration_seconds - 6) < 0.3, `duration ${probe.duration_seconds}`);
  assert.equal(probe.has_video, true);
  assert.equal(probe.has_audio, true);
  assert.ok(result.warnings.some((w: string) => /audio .* is cut/.test(w)));
  assert.ok((await call('list_library', { origin: 'assembled' })).items.some((i: any) => i.id === result.item.id));
});

withFfmpeg('assemble_video: crossfade, trims, and trim_to_audio hold the right length', async () => {
  const items = (await call('list_library', { origin: 'imported', batch: 'real' })).items;
  const clips = items.filter((i: any) => i.kind === 'video').sort((a: any, b: any) => a.name.localeCompare(b.name)).map((i: any) => i.id);
  const song = items.find((i: any) => i.kind === 'audio').id;

  const xfade = await assemble({ clips, audio: song, transition: 'crossfade', crossfade_seconds: 0.5, name: 'xfade' });
  assert.ok(Math.abs((await call('probe_media', { item_id: xfade.item.id })).duration_seconds - 5) < 0.3);

  const trimmed = await assemble({ clips: [{ item_id: clips[0], seconds: 1 }, clips[1]], name: 'trimmed' });
  const trimmedProbe = await call('probe_media', { item_id: trimmed.item.id });
  assert.ok(Math.abs(trimmedProbe.duration_seconds - 3) < 0.3);
  assert.equal(trimmedProbe.has_audio, false);

  const toAudio = await assemble({ clips, audio: song, end: 'trim_to_audio', name: 'toaudio' });
  const audioProbe = await call('probe_media', { item_id: toAudio.item.id });
  assert.ok(Math.abs(audioProbe.duration_seconds - 10) < 0.4, `held to ${audioProbe.duration_seconds}`);
  assert.ok(toAudio.warnings.some((w: string) => /last frame is held/.test(w)));

  const faded = await assemble({ clips, audio: song, end: 'fade_out', fade_seconds: 1, name: 'faded' });
  assert.ok(Math.abs((await call('probe_media', { item_id: faded.item.id })).duration_seconds - 6) < 0.3);
});

withFfmpeg('assemble_video runs in the background and can be polled and cancelled', async () => {
  const items = (await call('list_library', { origin: 'imported', batch: 'real' })).items;
  const clips = items.filter((i: any) => i.kind === 'video').map((i: any) => i.id);
  const started = await call('assemble_video', { clips, transition: 'crossfade', name: 'bg' });
  assert.equal(started.status, 'running');
  const done = await call('get_assembly', { assembly_id: started.id, wait_seconds: 60 });
  assert.equal(done.status, 'done');
  assert.equal(done.progress, 1);

  const another = await call('assemble_video', { clips, transition: 'crossfade', name: 'bg2' });
  const cancel = await call('cancel_job', { assembly_id: another.id });
  assert.equal(cancel.cancelled, true);
  const ended = await call('get_assembly', { assembly_id: another.id, wait_seconds: 30 });
  assert.ok(['cancelled', 'done'].includes(ended.status), `status ${ended.status}`);
});

withFfmpeg('assemble_video validates its inputs', async () => {
  const items = (await call('list_library', { origin: 'imported', batch: 'real' })).items;
  const clip = items.find((i: any) => i.kind === 'video').id;
  const song = items.find((i: any) => i.kind === 'audio').id;
  const still = items.find((i: any) => i.kind === 'image').id;
  await rejects(service.callTool('assemble_video', { clips: [] }), 'invalid_argument');
  await rejects(service.callTool('assemble_video', { clips: [still] }), 'invalid_argument');
  await rejects(service.callTool('assemble_video', { clips: [clip], audio: clip }), 'invalid_argument');
  await rejects(service.callTool('assemble_video', { clips: ['gen-99999'] }), 'not_found');
  await rejects(service.callTool('assemble_video', { clips: [clip], audio: song, transition: 'wipe' }), 'invalid_argument');
  await rejects(service.callTool('get_assembly', { assembly_id: 99999 }), 'not_found');
});

test('generate_image adds elements before the style, and a wrong kind says which argument to use', async () => {
  saveStyle(db, { name: 'Red coat', text: 'wearing a red trench coat', kind: 'element' });
  saveStyle(db, { name: 'Wide hat', text: 'wide-brim hat', kind: 'element' });
  const job = await call('generate_image', { prompt: 'a fox', style: '1930s movie poster', elements: ['red coat', 'Wide hat'] });
  assert.equal(job.prompt, 'a fox, wearing a red trench coat, wide-brim hat, bold lithograph, limited palette');
  assert.equal(job.style, '1930s movie poster + Red coat + Wide hat');
  released.shift()?.();
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });

  await assert.rejects(service.callTool('generate_image', { prompt: 'x', style: 'Red coat' }), /is an element: pass it as `elements`/);
  await assert.rejects(service.callTool('generate_image', { prompt: 'x', elements: ['1930s movie poster'] }), /is a style: pass it as `style`/);
  await assert.rejects(service.callTool('generate_image', { prompt: 'x', elements: ['nope'] }), (e: unknown) => e instanceof ApiError && e.code === 'not_found' && /"Red coat"/.test(e.message));
  await assert.rejects(service.callTool('generate_image', { prompt: 'x', elements: 'Red coat' }), /list of saved element names/);
  const listed = (await call('list_styles')).styles.map((s: { name: string; kind: string }) => [s.name, s.kind]);
  assert.deepEqual(listed, [['1930s movie poster', 'style'], ['Red coat', 'element'], ['Wide hat', 'element']]);
});
