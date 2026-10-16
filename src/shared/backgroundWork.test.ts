import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NO_BACKGROUND_WORK, STYLE_SAMPLES_BATCH, describeBackgroundWork, isBackgroundJob, summarizeBackgroundWork } from './backgroundWork';

const example = (status: 'queued' | 'running' | 'done') => ({ batch: STYLE_SAMPLES_BATCH, source: 'ui' as const, status });
const outside = (status: 'queued' | 'running' | 'done') => ({ batch: null, source: 'mcp' as const, status });
const own = (status: 'queued' | 'running' | 'done') => ({ batch: null, source: 'ui' as const, status });

test('only examples and outside requests are background jobs', () => {
  assert.equal(isBackgroundJob(example('queued')), true);
  assert.equal(isBackgroundJob(outside('queued')), true);
  assert.equal(isBackgroundJob(own('queued')), false);
});

test('summarizing counts the waiting and running ones and ignores finished and own jobs', () => {
  const work = summarizeBackgroundWork([example('running'), example('queued'), example('done'), outside('queued'), own('running')]);
  assert.deepEqual(work, { styleExamples: 2, outside: 1, running: 'styleExample' });
  assert.deepEqual(summarizeBackgroundWork([own('running')]), NO_BACKGROUND_WORK);
});

test('only an example already being made, or outside requests, are said to hold the queue up', () => {
  assert.equal(describeBackgroundWork({ styleExamples: 3, outside: 0, running: null }), null);
  assert.match(describeBackgroundWork({ styleExamples: 3, outside: 0, running: 'styleExample' }) ?? '', /style example/);
  assert.equal(describeBackgroundWork({ styleExamples: 0, outside: 2, running: 'outside' }), 'Waiting for 2 requests from outside the app');
  assert.equal(describeBackgroundWork({ styleExamples: 0, outside: 1, running: null }), '1 request from outside the app also in the queue');
});
