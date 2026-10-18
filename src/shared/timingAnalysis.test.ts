import assert from 'node:assert/strict';
import { test } from 'node:test';
import { filterByModels, groupByModel, modelOf } from './timingAnalysis';
import { TimingStatRow } from './types';

function row(id: number, modelName: string | null, estimateMs: number | null, actualMs: number): TimingStatRow {
  return {
    id,
    createdAt: '2026-01-01T00:00:00.000Z',
    family: 'z-image',
    modelName,
    kind: 'image',
    width: 1024,
    height: 1024,
    steps: 8,
    cfg: 1,
    length: null,
    warm: true,
    estimateMs,
    estimateGenerateMs: estimateMs,
    actualMs,
    loadMs: null,
    generateMs: actualMs,
    samplingMs: actualMs,
    finishMs: 0,
    samplerSteps: 8,
    paceMs: null,
  };
}

const rows = [row(4, 'Finetune', 1000, 1500), row(3, null, 1000, 1000), row(2, 'Finetune', 1000, 500), row(1, null, null, 900)];

test('a run is labelled by its saved variant, else the family\'s built-in model', () => {
  assert.equal(modelOf(rows[0]), 'Finetune');
  assert.equal(modelOf(rows[1]), 'Z Image Turbo');
});

test('runs are grouped per model, most-run first, each summarized on its own', () => {
  const groups = groupByModel(rows, 'total');
  assert.deepEqual(groups.map((g) => [g.model, g.runs]), [['Finetune', 2], ['Z Image Turbo', 2]]);
  assert.equal(groups[0].summary?.biasPct, 0); // +50% and -50%
  assert.equal(groups[1].summary?.runs, 1); // the run with no estimate is not compared
});

test('filtering keeps the chosen models, and an empty choice keeps every run', () => {
  assert.equal(filterByModels(rows, new Set()).length, 4);
  assert.deepEqual(filterByModels(rows, new Set(['Finetune'])).map((r) => r.id), [4, 2]);
  assert.deepEqual(filterByModels(rows, new Set(['Finetune', 'Z Image Turbo'])).length, 4);
  assert.equal(filterByModels(rows, new Set(['Nope'])).length, 0);
});
