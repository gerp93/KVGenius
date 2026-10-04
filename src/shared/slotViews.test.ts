import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SlotJob, batchShownFor, shouldFollow } from './slotViews';

const job = (batchId: number, slotId: string | undefined, status: SlotJob['status']): SlotJob => ({ batchId, slotId, status });

test('each tab shows its own batch, whatever finishes for the other', () => {
  const jobs = [job(1, 'a', 'done'), job(2, 'b', 'done'), job(3, 'a', 'done')];
  // Tab a was last pointed at batch 1; the newer batch 3 only shows once it is pointed there.
  assert.equal(batchShownFor(jobs, { a: 1, b: 2 }, 'a'), 1);
  assert.equal(batchShownFor(jobs, { a: 3, b: 2 }, 'b'), 2);
});

test('a tab with no results of its own shows nothing, even when others have some', () => {
  const jobs = [job(1, 'a', 'done')];
  assert.equal(batchShownFor(jobs, { a: 1 }, 'b'), null);
  assert.equal(batchShownFor([], {}, 'a'), null);
});

test('jobs with no tab belong to no tab', () => {
  const jobs = [job(1, undefined, 'done')];
  assert.equal(batchShownFor(jobs, {}, 'a'), null);
  assert.equal(batchShownFor(jobs, { a: 1 }, 'a'), null);
});

test('a viewed batch that is gone falls back to the tab\'s newest, not another tab\'s', () => {
  const jobs = [job(1, 'a', 'done'), job(2, 'b', 'done'), job(3, 'a', 'done')];
  assert.equal(batchShownFor(jobs, { a: 99 }, 'a'), 3);
  assert.equal(batchShownFor(jobs, {}, 'a'), 3);
});

test('a tab follows a new batch unless it is showing one still being worked on', () => {
  assert.equal(shouldFollow([], {}, 'a'), true);
  assert.equal(shouldFollow([job(1, 'a', 'done')], { a: 1 }, 'a'), true);
  assert.equal(shouldFollow([job(1, 'a', 'failed')], { a: 1 }, 'a'), true);
  assert.equal(shouldFollow([job(1, 'a', 'running'), job(1, 'a', 'queued')], { a: 1 }, 'a'), false);
});

test('another tab\'s work in progress does not stop this tab following', () => {
  const jobs = [job(1, 'a', 'done'), job(2, 'b', 'running')];
  assert.equal(shouldFollow(jobs, { a: 1, b: 2 }, 'a'), true);
  assert.equal(shouldFollow(jobs, { a: 1, b: 2 }, 'b'), false);
});
