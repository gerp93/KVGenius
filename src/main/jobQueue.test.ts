import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { JobQueue, JobRunner, MAX_QUEUED_JOBS } from './jobQueue';
import { JOBS_SCHEMA, getJob, insertJob, markJobRunning } from './jobStore';
import { JobRequest } from '../shared/jobs';

class CancelledError extends Error {}

function newDb(): DatabaseSync {
  const db = new DatabaseSync(':memory:');
  db.exec(JOBS_SCHEMA);
  return db;
}

function request(overrides: Partial<JobRequest> = {}): JobRequest {
  return {
    family: 'z-image',
    params: { prompt: 'a cat', width: 512, height: 512, seed: 1, steps: 4, cfg: 1 },
    source: 'mcp',
    ...overrides,
  };
}

/** A runner the test controls: each call parks until release()/fail() is called for it. */
function controlledRunner() {
  const calls: Array<{ id: number; release: () => void; fail: (e: Error) => void }> = [];
  let active = 0;
  let maxActive = 0;
  const runner: JobRunner = (job) => {
    active++;
    maxActive = Math.max(maxActive, active);
    return new Promise((resolve, reject) => {
      calls.push({
        id: job.id,
        release: () => {
          active--;
          resolve({ generationId: job.id * 100 });
        },
        fail: (e) => {
          active--;
          reject(e);
        },
      });
    });
  };
  return { runner, calls, maxActive: () => maxActive };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('runs jobs one at a time, in submission order', async () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  const a = q.submit(request());
  const b = q.submit(request());
  await tick();
  assert.equal(c.calls.length, 1);
  assert.equal(q.get(a.id)?.status, 'running');
  assert.equal(q.get(b.id)?.status, 'queued');

  c.calls[0].release();
  await q.wait(a.id);
  await tick();
  assert.equal(c.calls.length, 2);
  assert.equal(c.calls[1].id, b.id);
  c.calls[1].release();
  const done = await q.wait(b.id);
  assert.equal(done.status, 'done');
  assert.equal(done.generationId, b.id * 100);
  assert.equal(c.maxActive(), 1);
});

test('a failed job records its error and does not stop the next one', async () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  const a = q.submit(request());
  const b = q.submit(request());
  await tick();
  c.calls[0].fail(new Error('ComfyUI not reachable'));
  const failed = await q.wait(a.id);
  assert.equal(failed.status, 'failed');
  assert.equal(failed.error, 'ComfyUI not reachable');
  await tick();
  c.calls[1].release();
  assert.equal((await q.wait(b.id)).status, 'done');
});

test('cancelling a waiting job means it never runs', async () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  q.submit(request());
  const b = q.submit(request());
  await tick();
  assert.equal(await q.cancel(b.id), true);
  assert.equal(q.get(b.id)?.status, 'cancelled');
  c.calls[0].release();
  await tick();
  assert.equal(c.calls.length, 1);
});

test('cancelling the running job interrupts the backend and ends as cancelled', async () => {
  const c = controlledRunner();
  let interrupted = 0;
  const q = new JobQueue(newDb(), c.runner, {
    cancelRunning: async () => {
      interrupted++;
      c.calls[0].fail(new CancelledError('Generation cancelled.'));
    },
    isCancellation: (e) => e instanceof CancelledError,
  });
  const a = q.submit(request());
  await tick();
  assert.equal(await q.cancelRunning(), true);
  const job = await q.wait(a.id);
  assert.equal(job.status, 'cancelled');
  assert.equal(job.error, null);
  assert.equal(interrupted, 1);
});

test('cancel returns false for unknown or finished jobs', async () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  const a = q.submit(request());
  await tick();
  c.calls[0].release();
  await q.wait(a.id);
  assert.equal(await q.cancel(a.id), false);
  assert.equal(await q.cancel(9999), false);
  assert.equal(await q.cancelRunning(), false);
});

test('cancelQueued cancels only waiting jobs, optionally by batch', async () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  const running = q.submit(request({ batch: 'mv' }));
  const b1 = q.submit(request({ batch: 'mv' }));
  const other = q.submit(request({ batch: 'other' }));
  await tick();
  assert.equal(q.cancelQueued('mv'), 1);
  assert.equal(q.get(b1.id)?.status, 'cancelled');
  assert.equal(q.get(other.id)?.status, 'queued');
  assert.equal(q.get(running.id)?.status, 'running');
});

test('jobs left queued or running by a previous session become interrupted', () => {
  const db = newDb();
  const queued = insertJob(db, request());
  const running = insertJob(db, request());
  markJobRunning(db, running.id);
  const q = new JobQueue(db, async () => ({ generationId: 1 }), { cancelRunning: async () => undefined });
  assert.equal(q.get(queued.id)?.status, 'interrupted');
  assert.equal(q.get(running.id)?.status, 'interrupted');
  assert.ok(q.get(running.id)?.finishedAt);
});

test('list filters by batch and status, newest first', async () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  q.submit(request({ batch: 'a' }));
  const b = q.submit(request({ batch: 'b' }));
  q.submit(request({ batch: 'a' }));
  assert.deepEqual(q.list({ batch: 'a' }).map((j) => j.batch), ['a', 'a']);
  assert.equal(q.list({ batch: 'b' })[0].id, b.id);
  assert.ok(q.list({ status: 'queued' }).length >= 1);
  const all = q.list();
  assert.ok(all[0].id > all[all.length - 1].id);
});

test('params round-trip through the store', () => {
  const q = new JobQueue(newDb(), controlledRunner().runner, { cancelRunning: async () => undefined });
  const job = q.submit(request({ params: { prompt: 'x', width: 640, height: 480, seed: 7, steps: 4, cfg: 1, length: 81, sourceImagePath: '/a/b.png' } }));
  assert.deepEqual(q.get(job.id)?.params.sourceImagePath, '/a/b.png');
  assert.equal(q.get(job.id)?.params.length, 81);
});

test('the queue refuses new jobs once too many are waiting', () => {
  const c = controlledRunner();
  const q = new JobQueue(newDb(), c.runner, { cancelRunning: async () => undefined });
  q.submit(request()); // starts running
  for (let i = 0; i < MAX_QUEUED_JOBS; i++) q.submit(request());
  assert.throws(() => q.submit(request()), /queue is full/);
});

test('a job submitted or stored under the retired family key reads back under the current one', () => {
  const db = newDb();
  const stored = insertJob(db, request({ family: 'z-image-turbo' }));
  assert.equal(stored.family, 'z-image');
  db.exec("UPDATE jobs SET family = 'z-image-turbo'");
  assert.equal((db.prepare('SELECT family FROM jobs').get() as { family: string }).family, 'z-image-turbo');
  assert.equal(getJob(db, stored.id)?.family, 'z-image');
});
