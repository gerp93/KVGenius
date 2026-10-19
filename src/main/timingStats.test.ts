import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initDatabase, insertGeneration } from './db';
import { backfillTimingModels, insertTiming, listTimingRows } from './timingStats';
import { TimingStatRow } from '../shared/types';

const timing = (modelName: string | null): Omit<TimingStatRow, 'id' | 'createdAt'> => ({
  family: 'z-image', modelName, kind: 'image', width: 64, height: 64, steps: 4, cfg: 1, length: null, warm: false,
  estimateMs: 1000, estimateGenerateMs: 1000, actualMs: 1200, loadMs: null, generateMs: 1200, samplingMs: 1000, finishMs: 200,
  samplerSteps: 4, paceMs: 250,
});
const params = { prompt: 'p', width: 64, height: 64, seed: 1, steps: 4, cfg: 1 };

test('a timing keeps the model it was made with', () => {
  const db = initDatabase(':memory:');
  insertTiming(db, timing('Finetune'));
  insertTiming(db, timing(null));
  assert.deepEqual(listTimingRows(db).map((r) => r.modelName), [null, 'Finetune']);
});

test('older timings take their model from their generation, and nothing else is touched', () => {
  const db = initDatabase(':memory:');
  const named = insertTiming(db, timing(null));
  const builtIn = insertTiming(db, timing(null));
  const orphan = insertTiming(db, timing(null));
  const keeps = insertTiming(db, timing('Own name'));
  const variant = { ...params, modelName: 'Finetune', modelSettings: { files: { unet: 'a.safetensors' } } };
  insertGeneration(db, variant, 'z-image', 'a.png', named);
  insertGeneration(db, params, 'z-image', 'b.png', builtIn);
  insertGeneration(db, variant, 'z-image', 'c.png', keeps);

  backfillTimingModels(db);
  backfillTimingModels(db); // running it again changes nothing

  const byId = new Map(listTimingRows(db).map((r) => [r.id, r.modelName]));
  assert.equal(byId.get(named), 'Finetune');
  assert.equal(byId.get(builtIn), null);
  assert.equal(byId.get(orphan), null);
  assert.equal(byId.get(keeps), 'Own name');
});
