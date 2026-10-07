import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ModelProfileInput,
  isSafeModelFileName,
  parseModelSettings,
  profileMatchesSettings,
  profileSettings,
  serializeModelSettings,
  validateProfileInput,
} from './modelProfiles';

function input(overrides: Partial<ModelProfileInput> = {}): ModelProfileInput {
  return {
    family: 'z-image',
    name: 'Photoreal',
    files: { diffusionModel: 'photoreal.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' },
    sampler: { steps: 30, cfg: 4, sampler: 'euler', scheduler: 'simple', shift: 3 },
    ...overrides,
  };
}

test('a complete profile is accepted and trimmed', () => {
  const r = validateProfileInput(input({ name: '  Photoreal  ' }));
  assert.equal(r.ok, true);
  if (r.ok) assert.equal(r.value.name, 'Photoreal');
});

test('each missing or bad piece is named', () => {
  const bad = (overrides: Partial<ModelProfileInput>) => {
    const r = validateProfileInput(input(overrides));
    return r.ok ? '' : r.message;
  };
  assert.match(bad({ family: 'nope' }), /kind of model/);
  assert.match(bad({ name: '   ' }), /name/);
  assert.match(bad({ name: 'x'.repeat(61) }), /60 characters/);
  assert.match(bad({ name: 'z image turbo' }), /built-in/);
  assert.match(bad({ files: { diffusionModel: '', textEncoder: 'a', vae: 'b' } }), /image model/i);
  assert.match(bad({ files: { diffusionModel: '../x.safetensors', textEncoder: 'a', vae: 'b' } }), /not a valid file name/);
  assert.match(bad({ sampler: { steps: 0, cfg: 1, sampler: 'euler', scheduler: 'simple', shift: 3 } }), /Steps/);
  assert.match(bad({ sampler: { steps: 8.5, cfg: 1, sampler: 'euler', scheduler: 'simple', shift: 3 } }), /Steps/);
  assert.match(bad({ sampler: { steps: 8, cfg: 99, sampler: 'euler', scheduler: 'simple', shift: 3 } }), /CFG/);
  assert.match(bad({ sampler: { steps: 8, cfg: 1, sampler: '', scheduler: 'simple', shift: 3 } }), /sampler/);
});

test('file names cannot leave the models folder', () => {
  assert.equal(isSafeModelFileName('model.safetensors'), true);
  assert.equal(isSafeModelFileName('sub/dir/model.safetensors'), true);
  for (const bad of ['', '../m.safetensors', 'a/../../m', '/etc/passwd', 'C:\\x.safetensors', '\\\\server\\x', 'a//b']) {
    assert.equal(isSafeModelFileName(bad), false, bad);
  }
});

test('settings serialize the same whatever order the keys came in', () => {
  const a = serializeModelSettings({ files: { vae: 'v', diffusionModel: 'd' }, sampler: 's', scheduler: 'c', shift: 3 });
  const b = serializeModelSettings({ files: { diffusionModel: 'd', vae: 'v' }, sampler: 's', scheduler: 'c', shift: 3 });
  assert.equal(a, b);
  assert.equal(serializeModelSettings(null), null);
  assert.deepEqual(parseModelSettings(a), { files: { diffusionModel: 'd', vae: 'v' }, sampler: 's', scheduler: 'c', shift: 3 });
  assert.equal(parseModelSettings('not json'), null);
  assert.equal(parseModelSettings(null), null);
});

test('a profile matches a record only while its files and sampler are unchanged', () => {
  const r = validateProfileInput(input());
  assert.ok(r.ok);
  if (!r.ok) return;
  const profile = r.value;
  const made = profileSettings(profile);
  assert.equal(profileMatchesSettings(profile, made), true);
  assert.equal(profileMatchesSettings({ ...profile, files: { ...profile.files, vae: 'other.safetensors' } }, made), false);
  assert.equal(profileMatchesSettings({ ...profile, sampler: { ...profile.sampler, scheduler: 'karras' } }, made), false);
  assert.equal(profileMatchesSettings(profile, null), false);
});

test('a video profile carries files only: sampler input is ignored and not required', () => {
  const files = {
    highNoiseModel: 'hi.safetensors',
    lowNoiseModel: 'lo.safetensors',
    textEncoder: 'umt5.safetensors',
    vae: 'v.safetensors',
    highNoiseLora: 'lh.safetensors',
    lowNoiseLora: 'll.safetensors',
  };
  const r = validateProfileInput({ family: 'wan22-i2v', name: 'Wan custom', files, sampler: { steps: 0, cfg: 99, sampler: '', scheduler: '', shift: -1 } });
  assert.ok(r.ok);
  if (!r.ok) return;
  assert.deepEqual(profileSettings(r.value), { files });
  // every one of the six files is needed
  const { lowNoiseLora: _omitted, ...five } = files;
  const missing = validateProfileInput({ family: 'wan22-i2v', name: 'Wan custom', files: five, sampler: r.value.sampler });
  assert.equal(missing.ok, false);
  // the built-in video name is taken
  const clash = validateProfileInput({ family: 'wan22-i2v', name: 'wan 2.2 IMAGE to video', files, sampler: r.value.sampler });
  assert.equal(clash.ok, false);
  // a video profile's serialized settings carry no sampler, and read back the same
  assert.deepEqual(parseModelSettings(serializeModelSettings(profileSettings(r.value))), { files });
});
