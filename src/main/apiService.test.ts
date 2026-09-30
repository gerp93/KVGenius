import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { initDatabase, insertGeneration } from './db';
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
      const file = path.join(dir, `out-${job.id}.${job.family === 'wan22-i2v' ? 'mp4' : 'png'}`);
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
  assert.deepEqual(caps.families.map((f: any) => f.family), ['z-image-turbo', 'wan22-i2v']);
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

test('generate_video validates the source item', async () => {
  await rejects(service.callTool('generate_video', { prompt: 'x', source: 'imp-9999' }), 'not_found');
  await rejects(service.callTool('generate_video', { prompt: 'x', source: 'garbage' }), 'not_found');
  const song = (await call('list_library', { kind: 'audio' })).items[0];
  await rejects(service.callTool('generate_video', { prompt: 'x', source: song.id }), 'invalid_argument');
  await rejects(service.callTool('generate_video', { prompt: 'x' }), 'invalid_argument');
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
  const uiJob = queue.submit({ family: 'z-image-turbo', params: uiParams, source: 'ui', batch: 'shared' });
  released.shift()?.();
  const uiDone = await queue.wait(uiJob.id);
  assert.equal(uiDone.status, 'done');
  const uiGen = `gen-${uiDone.generationId}`;

  // One from before jobs were recorded at all (no job row).
  const legacyFile = path.join(dir, 'legacy.png');
  fs.writeFileSync(legacyFile, 'x');
  const legacyGen = `gen-${insertGeneration(db, { ...uiParams, prompt: 'SECRET legacy prompt' }, 'z-image-turbo', legacyFile, null).id}`;

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
  const waitingUi = queue.submit({ family: 'z-image-turbo', params: uiParams, source: 'ui', batch: 'shared' });
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
