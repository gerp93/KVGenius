import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { JobQueue } from './jobQueue';
import { JOBS_SCHEMA } from './jobStore';
import { ModelTestDeps, explainTestFailure, runModelTest } from './modelTest';
import { describeExecutionError } from './comfyui';
import { ModelProfileInput } from '../shared/modelProfiles';
import { GenerationParams } from '../shared/types';

const input: ModelProfileInput = {
  family: 'z-image',
  name: '',
  files: { diffusionModel: 'photoreal.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' },
  sampler: { steps: 30, cfg: 4, sampler: 'euler', scheduler: 'karras', shift: 3 },
};

function deps(generate: ModelTestDeps['generate']): ModelTestDeps {
  return { runExclusive: (work) => work(), generate };
}

test('a test render uses the profile as set up, small and short, and returns the picture', async () => {
  let seen: { family: string; params: GenerationParams } | null = null;
  const result = await runModelTest(
    input,
    deps(async (family, params) => {
      seen = { family, params };
      return { bytes: Buffer.from('png-bytes'), extension: '.png' };
    }),
  );
  assert.equal(result.ok, true);
  assert.equal(result.imageBase64, Buffer.from('png-bytes').toString('base64'));
  assert.equal(result.mime, 'image/png');
  assert.match(result.message, /256x256.*8 steps/);
  assert.ok(seen);
  const s = seen as unknown as { family: string; params: GenerationParams };
  assert.equal(s.family, 'z-image');
  assert.equal(s.params.steps, 8);
  assert.equal(s.params.cfg, 4);
  assert.equal(s.params.width, 256);
  assert.equal(s.params.modelSettings?.files.diffusionModel, 'photoreal.safetensors');
  assert.equal(s.params.modelSettings?.scheduler, 'karras');
});

test('fewer steps than the cap are kept', async () => {
  let steps = 0;
  await runModelTest({ ...input, sampler: { ...input.sampler, steps: 4 } }, deps(async (_f, p) => ((steps = p.steps), { bytes: Buffer.from('x'), extension: '.png' })));
  assert.equal(steps, 4);
});

test('bad input and failures come back as a message, never a throw', async () => {
  const never = deps(async () => {
    throw new Error('should not run');
  });
  const bad = await runModelTest({ ...input, files: {} }, never);
  assert.equal(bad.ok, false);
  assert.match(bad.message, /file/i);
  const failed = await runModelTest(input, deps(async () => { throw new Error('ComfyUI could not run it: size mismatch for x (in UNETLoader)'); }));
  assert.equal(failed.ok, false);
  assert.match(failed.message, /do not fit together/);
  const busy = await runModelTest(input, { runExclusive: async () => { throw new Error('The queue is busy - wait for it to finish, then try again.'); }, generate: async () => ({ bytes: Buffer.alloc(0), extension: '.png' }) });
  assert.match(busy.message, /queue is busy/);
});

test('ComfyUI and network messages are put in plain words', () => {
  assert.match(explainTestFailure('ComfyUI not reachable at http://localhost:8000: TypeError: fetch failed'), /not reachable/);
  assert.match(explainTestFailure('ComfyUI returned 400 Bad Request: {"error":{"type":"prompt_outputs_failed_validation"},"node_errors":{"57:28":{"errors":[{"type":"value_not_in_list"}]}}}'), /does not list one of the chosen files/);
  assert.match(explainTestFailure('CUDA out of memory. Tried to allocate 2 GiB'), /out of memory/);
  assert.equal(explainTestFailure('something odd'), 'something odd');
  assert.match(explainTestFailure(''), /gave no reason/);
});

test("a failed ComfyUI run's own reason is read from its history", () => {
  const failed = { status: { status_str: 'error', messages: [['execution_start', {}], ['execution_error', { exception_message: 'size mismatch for blocks.0', node_type: 'UNETLoader' }]] } };
  assert.equal(describeExecutionError(failed), 'size mismatch for blocks.0 (in UNETLoader)');
  assert.match(String(describeExecutionError({ status: { status_str: 'error', messages: [] } })), /no reason/);
  assert.equal(describeExecutionError({ status: { status_str: 'success', messages: [] } }), null);
  assert.equal(describeExecutionError({}), null);
});

/** Waits (up to 2 s) for something that happens on its own schedule, instead of guessing a delay. */
async function until(condition: () => boolean): Promise<void> {
  const deadline = Date.now() + 2000;
  while (!condition()) {
    if (Date.now() > deadline) throw new Error('timed out waiting for the queue');
    await new Promise((r) => setTimeout(r, 5));
  }
}

test('runExclusive refuses while a job runs, holds the queue while it works, and restarts it after', async () => {
  const db = new DatabaseSync(':memory:');
  db.exec(JOBS_SCHEMA);
  const order: string[] = [];
  const releases = new Map<number, () => void>();
  const queue = new JobQueue(
    db,
    (job) =>
      new Promise((resolve) => {
        order.push(`run ${job.id}`);
        releases.set(job.id, () => resolve({ generationId: job.id }));
      }),
    { cancelRunning: async () => undefined },
  );
  const request = { family: 'z-image', params: { prompt: 'a', width: 64, height: 64, seed: 1, steps: 1, cfg: 1 }, source: 'ui' as const };
  queue.submit(request);
  await until(() => order.includes('run 1'));
  await assert.rejects(() => queue.runExclusive(async () => 1), /queue is busy/);
  releases.get(1)?.();
  await until(() => queue.get(1)?.status === 'done');

  let finishWork: () => void = () => undefined;
  const work = queue.runExclusive(
    () =>
      new Promise<string>((resolve) => {
        order.push('exclusive start');
        finishWork = () => resolve('done');
      }),
  );
  queue.submit(request);
  await until(() => order.includes('exclusive start'));
  // Give a wrongly started job every chance to show up before checking that it did not.
  await new Promise((r) => setTimeout(r, 50));
  assert.deepEqual(order, ['run 1', 'exclusive start'], 'the job submitted during the exclusive work waits');
  finishWork();
  assert.equal(await work, 'done');
  await until(() => order.includes('run 2'));
  assert.deepEqual(order, ['run 1', 'exclusive start', 'run 2']);
  releases.get(2)?.();
  await until(() => queue.get(2)?.status === 'done');
});
