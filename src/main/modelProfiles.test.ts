import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findDuplicateGeneration, getGenerationById, initDatabase, insertGeneration } from './db';
import { deleteModelProfile, findModelProfileByName, getModelProfile, listModelProfiles, saveModelProfile } from './modelProfiles';
import { ModelProfileInput, ModelSettings, profileSettings } from '../shared/modelProfiles';

function profile(name: string, overrides: Partial<ModelProfileInput> = {}): ModelProfileInput {
  return {
    family: 'z-image',
    name,
    files: { diffusionModel: `${name}.safetensors`, textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' },
    sampler: { steps: 30, cfg: 4, sampler: 'euler', scheduler: 'simple', shift: 3 },
    ...overrides,
  };
}

const params = { prompt: 'a fox', width: 64, height: 64, seed: 1, steps: 30, cfg: 4 };

test('profiles are saved, listed alphabetically, found by name ignoring case, edited and deleted', () => {
  const db = initDatabase(':memory:');
  const b = saveModelProfile(db, profile('Bravo'));
  saveModelProfile(db, profile('alpha'));
  assert.deepEqual(listModelProfiles(db).map((p) => p.name), ['alpha', 'Bravo']);
  assert.equal(findModelProfileByName(db, ' BRAVO ')?.id, b.id);
  assert.deepEqual(getModelProfile(db, b.id)?.files, profile('Bravo').files);

  const edited = saveModelProfile(db, profile('Bravo', { sampler: { steps: 12, cfg: 2, sampler: 'euler', scheduler: 'karras', shift: 1 } }), b.id);
  assert.equal(edited.sampler.steps, 12);
  assert.equal(edited.sampler.scheduler, 'karras');

  deleteModelProfile(db, b.id);
  assert.equal(getModelProfile(db, b.id), null);
});

test('names are unique ignoring case, and bad input is refused with a reason', () => {
  const db = initDatabase(':memory:');
  saveModelProfile(db, profile('Photoreal'));
  assert.throws(() => saveModelProfile(db, profile('PHOTOREAL')), /already exists/);
  assert.throws(() => saveModelProfile(db, profile('x', { files: {} })), /file/i);
  assert.throws(() => saveModelProfile(db, profile('y'), 9999), /no longer exists/);
});

test('a generation keeps the model it was made with, whatever happens to the profile', () => {
  const db = initDatabase(':memory:');
  const saved = saveModelProfile(db, profile('Photoreal'));
  const settings = profileSettings(saved);
  const record = insertGeneration(db, { ...params, modelName: saved.name, modelSettings: settings }, 'z-image', '/out/a.png');
  deleteModelProfile(db, saved.id);
  const back = getGenerationById(db, record.id);
  assert.equal(back?.modelName, 'Photoreal');
  assert.deepEqual(back?.modelSettings, settings);
});

test('a generation made with the shipped template carries no model', () => {
  const db = initDatabase(':memory:');
  const record = insertGeneration(db, params, 'z-image', '/out/a.png');
  const back = getGenerationById(db, record.id);
  assert.equal(back?.modelName, null);
  assert.equal(back?.modelSettings, null);
  // a label without settings is not kept - the settings are what make it a different model
  const labelled = insertGeneration(db, { ...params, seed: 2, modelName: 'Ghost' }, 'z-image', '/out/b.png');
  assert.equal(getGenerationById(db, labelled.id)?.modelName, null);
});

test('the same seed and settings on a different model is not a duplicate', () => {
  const db = initDatabase(':memory:');
  const a: ModelSettings = profileSettings({ family: 'z-image', files: { diffusionModel: 'a.safetensors', textEncoder: 't', vae: 'v' }, sampler: { steps: 30, cfg: 4, sampler: 'euler', scheduler: 'simple', shift: 3 } });
  const b: ModelSettings = { ...a, files: { ...a.files, diffusionModel: 'b.safetensors' } };
  insertGeneration(db, { ...params, modelName: 'A', modelSettings: a }, 'z-image', '/out/a.png');
  const query = { prompt: 'a fox', width: 64, height: 64, seed: 1, steps: 30, cfg: 4 };
  assert.ok(findDuplicateGeneration(db, 'z-image', { ...query, modelSettings: a }));
  assert.equal(findDuplicateGeneration(db, 'z-image', { ...query, modelSettings: b }), null);
  assert.equal(findDuplicateGeneration(db, 'z-image', query), null);
  insertGeneration(db, params, 'z-image', '/out/plain.png');
  assert.ok(findDuplicateGeneration(db, 'z-image', query));
});

test('the generations table has the model columns', () => {
  const db = initDatabase(':memory:');
  const cols = (db.prepare('PRAGMA table_info(generations)').all() as unknown as { name: string }[]).map((c) => c.name);
  assert.ok(cols.includes('model_name') && cols.includes('model_settings'));
});
