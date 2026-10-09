import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initDatabase, insertGeneration, getGenerationById, setGenerationHidden } from './db';
import { deleteStyle, getStyle, listStyles, saveStyle } from './styles';
import { StyleSampleService, StyleSamplesQueue, getSampleSettings, saveSampleSettings, sampleRequest } from './styleSamples';
import { JobInfo, JobRequest } from '../shared/jobs';
import { BASELINE_ID, DEFAULT_SAMPLE_SETTINGS, SAMPLE_SIZE } from '../shared/styleSamples';

function harness() {
  const db = initDatabase(':memory:');
  const jobs = new Map<number, JobInfo>();
  const waiters = new Map<number, (job: JobInfo) => void>();
  const submitted: JobRequest[] = [];
  const trashed: number[] = [];
  let nextJob = 1;
  let nextFile = 1;
  const queue: StyleSamplesQueue = {
    submit(request) {
      submitted.push(request);
      const job: JobInfo = { id: nextJob++, source: request.source, batch: request.batch ?? null, family: request.family, params: request.params, status: 'queued', error: null, generationId: null, createdAt: '', startedAt: null, finishedAt: null };
      jobs.set(job.id, job);
      return job;
    },
    get: (id) => jobs.get(id) ?? null,
    wait: (id) => new Promise((resolve) => waiters.set(id, resolve)),
  };
  const service = new StyleSampleService({
    db,
    queue,
    trash: (ids) => void trashed.push(...ids),
    hide: (id) => setGenerationHidden(db, id, true),
    getRecord: (id) => {
      const record = getGenerationById(db, id);
      return record && trashed.includes(id) ? { ...record, trashedAt: 'now' } : record;
    },
    imageUrlFor: (p) => `url:${p}`,
    listStyles: () => listStyles(db),
    getStyle: (id) => getStyle(db, id),
  });
  /** Completes the oldest waiting job, making a Library record for it like the real runner. */
  const finish = async (jobId: number, status: 'done' | 'failed' = 'done') => {
    const job = jobs.get(jobId) as JobInfo;
    if (status === 'done') job.generationId = insertGeneration(db, job.params, job.family, `/out/images/s${nextFile++}.png`).id;
    else job.error = 'ComfyUI is not reachable';
    job.status = status;
    waiters.get(jobId)?.(job);
    await Promise.resolve();
    await Promise.resolve();
  };
  return { db, service, submitted, trashed, finish };
}

test('the standard prompt and seed default, can be saved, and are checked', () => {
  const db = initDatabase(':memory:');
  assert.deepEqual(getSampleSettings(db), DEFAULT_SAMPLE_SETTINGS);
  assert.deepEqual(saveSampleSettings(db, { prompt: ' a fox ', seed: 5 }), { prompt: 'a fox', seed: 5 });
  assert.deepEqual(getSampleSettings(db), { prompt: 'a fox', seed: 5 });
  assert.throws(() => saveSampleSettings(db, { prompt: '', seed: 5 }), /standard prompt/);
  assert.throws(() => saveSampleSettings(db, { prompt: 'x', seed: -3 }), /seed/);
});

test('an example is the standard prompt plus the wording, at the fixed seed and size', () => {
  const db = initDatabase(':memory:');
  const coat = saveStyle(db, { name: 'Coat', text: 'red coat', kind: 'element' });
  const settings = { prompt: 'a fox', seed: 9 };
  const request = sampleRequest(settings, coat);
  assert.equal(request.params.prompt, 'a fox, red coat');
  assert.equal(request.params.seed, 9);
  assert.equal(request.params.width, SAMPLE_SIZE);
  assert.equal(request.params.styleName, 'Coat');
  assert.equal(sampleRequest(settings, null).params.prompt, 'a fox');
  assert.equal(sampleRequest(settings, null).params.styleName, undefined);
});

test('the date the wording changed moves only when the wording does', () => {
  const db = initDatabase(':memory:');
  const made = saveStyle(db, { name: 'Noir', text: 'black and white' });
  assert.equal(made.textChangedAt, made.createdAt);
  db.prepare("UPDATE styles SET text_changed_at = '2020-01-01T00:00:00.000Z' WHERE id = ?").run(made.id);
  assert.equal(saveStyle(db, { name: 'Noir 2', text: 'black and white', kind: 'element' }, made.id).textChangedAt, '2020-01-01T00:00:00.000Z', 'a rename or kind change is not a new wording');
  assert.notEqual(saveStyle(db, { name: 'Noir 2', text: 'black and white, grainy' }, made.id).textChangedAt, '2020-01-01T00:00:00.000Z');
});

test('examples start as none, render through the queue, and go outdated when the wording changes', async () => {
  const { db, service, submitted, finish } = harness();
  const noir = saveStyle(db, { name: 'Noir', text: 'black and white' });
  assert.deepEqual(service.view().samples.map((s) => [s.id, s.state]), [[BASELINE_ID, 'none'], [noir.id, 'none']]);

  assert.equal(service.render([BASELINE_ID, noir.id]), 2);
  assert.deepEqual(submitted.map((r) => r.params.prompt), [DEFAULT_SAMPLE_SETTINGS.prompt, `${DEFAULT_SAMPLE_SETTINGS.prompt}, black and white`]);
  assert.deepEqual(service.view().samples.map((s) => s.state), ['queued', 'queued']);
  assert.equal(service.render([noir.id]), 0, 'one already waiting is not queued twice');

  await finish(1);
  assert.deepEqual(service.view().samples.map((s) => s.state), ['current', 'queued']);
  await finish(2);
  const view = service.view();
  assert.deepEqual(view.samples.map((s) => s.state), ['current', 'current']);
  assert.match(view.samples[1].imageUrl ?? '', /^url:\/out\/images\/s2\.png$/);
  assert.equal(getGenerationById(db, 2)?.hidden, true, 'examples stay out of the Library lists');

  saveStyle(db, { name: 'Noir', text: 'black and white, grainy' }, noir.id);
  assert.deepEqual(service.view().samples.map((s) => s.state), ['current', 'outdated']);
  assert.ok(service.view().samples[1].imageUrl, 'the old picture stays until the new one is made');

  saveSampleSettings(db, { prompt: 'a cat', seed: 1 });
  assert.deepEqual(service.view().samples.map((s) => s.state), ['outdated', 'outdated'], 'changing the standard prompt outdates everything');
});

test('a new example replaces the old one, which goes to the Trash; a failure is reported and can be retried', async () => {
  const { db, service, trashed, finish } = harness();
  const noir = saveStyle(db, { name: 'Noir', text: 'black and white' });
  service.render([noir.id]);
  await finish(1);
  saveStyle(db, { name: 'Noir', text: 'black and white, grainy' }, noir.id);
  service.render([noir.id]);
  await finish(2);
  assert.deepEqual(trashed, [1]);
  assert.equal(service.view().samples[1].state, 'current');

  saveStyle(db, { name: 'Noir', text: 'sepia' }, noir.id);
  service.render([noir.id]);
  await finish(3, 'failed');
  const failed = service.view().samples[1];
  assert.equal(failed.state, 'failed');
  assert.equal(failed.error, 'ComfyUI is not reachable');
  assert.equal(service.render([noir.id]), 1, 'a failed one can be queued again');
});

test('a picture sent to the Trash is no example, and deleting a style takes its example along', async () => {
  const { db, service, trashed, finish } = harness();
  const noir = saveStyle(db, { name: 'Noir', text: 'black and white' });
  service.render([noir.id]);
  await finish(1);
  trashed.push(1);
  assert.equal(service.view().samples[1].state, 'none');
  assert.equal(service.view().samples[1].imageUrl, null);

  trashed.length = 0;
  service.render([noir.id]);
  await finish(2);
  trashed.length = 0;
  deleteStyle(db, noir.id);
  service.forget(noir.id);
  assert.deepEqual(trashed, [2]);
  assert.deepEqual(service.view().samples.map((s) => s.id), [BASELINE_ID]);
});
