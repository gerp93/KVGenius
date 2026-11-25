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
import { ApiService } from './apiService';
import { TOOLS } from '../shared/tools';

/** list_styles and generate_image with an element that carries a LoRA. Apart from apiService.test.ts, with its own database and queue. */
let dir: string;
let db: DatabaseSync;
let service: ApiService;
let queue: JobQueue;
const runnerFamilies: string[] = [];

const fakeRunner: JobRunner = async (job) => {
  runnerFamilies.push(job.family);
  const file = path.join(dir, `out-${job.id}.png`);
  fs.writeFileSync(file, 'x');
  return { generationId: insertGeneration(db, job.params, job.family, file, null).id };
};

before(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-api-lora-'));
  db = initDatabase(':memory:');
  queue = new JobQueue(db, fakeRunner, { cancelRunning: async () => undefined });
  const assemblies = new AssemblyManager(db, { ffmpeg: () => null, outputDir: () => path.join(dir, 'videos'), tempDir: () => path.join(dir, 'tmp') });
  service = new ApiService({
    db,
    queue,
    assemblies,
    ffmpeg: () => null,
    comfyAvailable: async () => true,
    imageSize: () => ({ width: 1600, height: 900 }),
    imagePreviewFallback: async () => null,
    workspaceId: () => 1,
    setFavorite: () => '',
  });
  saveStyle(db, { name: "Twi'lek", text: "ffl-twi'lek-mk.1, blue skin", kind: 'element', lora: { file: 'ffl-twilek.safetensors', strength: 0.8, families: ['flux-dev'] } }, null, 1);
  saveStyle(db, { name: 'Red coat', text: 'wearing a red trench coat', kind: 'element' }, null, 1);
  saveStyle(db, { name: 'Noir', text: 'black and white', kind: 'style' }, null, 1);
});

after(() => fs.rmSync(dir, { recursive: true, force: true }));

const call = async (tool: string, args?: unknown) => (await service.callTool(tool, args)).data as any;
const jobParams = (jobId: number) => queue.get(jobId)!.params;

test('list_styles shows an element\'s LoRA file and strength, and nothing for the rest', async () => {
  const { styles } = await call('list_styles');
  assert.deepEqual(styles, [
    { name: 'Noir', kind: 'style', text: 'black and white' },
    { name: 'Red coat', kind: 'element', text: 'wearing a red trench coat' },
    { name: "Twi'lek", kind: 'element', text: "ffl-twi'lek-mk.1, blue skin", lora: { file: 'ffl-twilek.safetensors', strength: 0.8 } },
  ]);
});

test('generate_image with flux-dev carries the element\'s LoRA in the job, in the order the elements were named', async () => {
  const job = await call('generate_image', { prompt: 'a dancer', family: 'flux-dev', elements: ['Red coat', "twi'lek"] });
  assert.equal(job.notes, undefined);
  assert.equal(job.prompt, "a dancer, wearing a red trench coat, ffl-twi'lek-mk.1, blue skin");
  assert.deepEqual(jobParams(job.job_id).loras, [{ file: 'ffl-twilek.safetensors', strength: 0.8 }]);
  const done = await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
  assert.equal(done.status, 'done');
});

test('generate_image with a family that cannot load LoRAs leaves it out and says so, still adding the words', async () => {
  const job = await call('generate_image', { prompt: 'a dancer', elements: ["Twi'lek"] });
  assert.equal(job.family, 'z-image');
  assert.equal(job.prompt, "a dancer, ffl-twi'lek-mk.1, blue skin");
  assert.equal(jobParams(job.job_id).loras, undefined);
  assert.equal(job.notes.length, 1);
  assert.match(job.notes[0], /Twi'lek/);
  assert.match(job.notes[0], /Flux dev/);
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
});

test('an element without a LoRA adds no loras and no notes, and the tool descriptions mention them', async () => {
  const job = await call('generate_image', { prompt: 'a dancer', family: 'flux-dev', elements: ['Red coat'] });
  assert.equal(jobParams(job.job_id).loras, undefined);
  assert.equal(job.notes, undefined);
  await call('get_job', { job_id: job.job_id, wait_seconds: 5 });
  const text = (name: string) => JSON.stringify(TOOLS.find((t) => t.name === name));
  assert.match(text('generate_image'), /LoRA/);
  assert.match(text('list_styles'), /LoRA/);
});
