import { test } from 'node:test';
import assert from 'node:assert/strict';
import { V2V_STEPS, v2vSchedule } from '../shared/videoToVideo';
import { WAN22_V2V_NODE_MAP, fillVideoToVideo } from './videoToVideoPatch';
import v2vTemplate from './templates/wan22-v2v.json';
import t2vTemplate from './templates/wan22-t2v.json';
import { keepJobSources } from './sourceImages';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const fresh = () => JSON.parse(JSON.stringify(v2vTemplate)) as Template;
const params = (over: Record<string, unknown> = {}) => ({ prompt: 'a fox', width: 640, height: 384, seed: 77, steps: 4, cfg: 1, length: 49, denoise: 0.7, sourceVideoPath: '/v/a.mp4', ...over });

test('the step counts the schedule assumes are the template\'s own', () => {
  const t = v2vTemplate as unknown as Template;
  assert.equal(t['129:118'].inputs.value, V2V_STEPS.fast.steps);
  assert.equal(t['129:124'].inputs.value, V2V_STEPS.fast.split);
  assert.equal(t['129:128'].inputs.value, V2V_STEPS.high.steps);
  assert.equal(t['129:127'].inputs.value, V2V_STEPS.high.split);
});

test('the video to video graph is the text to video graph with the empty latent swapped for the encoded source', () => {
  const v2v = v2vTemplate as unknown as Template;
  const t2v = t2vTemplate as unknown as Template;
  assert.equal(v2v['129:98'], undefined, 'no empty latent');
  for (const id of Object.keys(t2v)) {
    if (id === '129:98') continue;
    assert.equal(v2v[id]?.class_type, t2v[id].class_type, `${id} is kept`);
  }
  assert.deepEqual(v2v['129:86'].inputs.latent_image, ['v2v-encode', 0]);
  assert.deepEqual(v2v['129:94'].inputs.fps, ['v2v-parts', 2], 'the result plays at the source frame rate');
  for (const id of Object.values(WAN22_V2V_NODE_MAP)) assert.ok(v2v[id], `${id} exists`);
});

test('strength decides where in the schedule sampling starts', () => {
  // Fast: 4 steps, hand over at 2
  assert.deepEqual(v2vSchedule(1, 1), { firstStart: 0, secondAddsNoise: false, secondStart: null }, 'from pure noise, like text to video');
  assert.deepEqual(v2vSchedule(0.7, 1), { firstStart: 1, secondAddsNoise: false, secondStart: null });
  assert.deepEqual(v2vSchedule(0.5, 1), { firstStart: 2, secondAddsNoise: true, secondStart: 2 });
  assert.deepEqual(v2vSchedule(0.25, 1), { firstStart: 2, secondAddsNoise: true, secondStart: 3 });
  assert.equal(v2vSchedule(0.05, 1).secondStart, 3, 'always at least one step is sampled');
  // High: 20 steps, hand over at 10
  assert.deepEqual(v2vSchedule(0.8, 3.5), { firstStart: 4, secondAddsNoise: false, secondStart: null });
  assert.deepEqual(v2vSchedule(0.3, 3.5), { firstStart: 10, secondAddsNoise: true, secondStart: 14 });
  // junk falls back to the default strength
  assert.deepEqual(v2vSchedule('lots', 1), v2vSchedule(0.7, 1));
});

test('filling the template sets the source, size, frames, prompt, seed and start step', () => {
  const w = fresh();
  fillVideoToVideo(w, params(), 'clip.mp4', 'wan22-v2v');
  assert.equal(w['v2v-load'].inputs.file, 'clip.mp4');
  assert.equal(w['v2v-scale'].inputs.width, 640);
  assert.equal(w['v2v-scale'].inputs.height, 384);
  assert.equal(w['v2v-frames'].inputs.length, 49);
  assert.equal(w['129:93'].inputs.text, 'a fox');
  assert.equal(w['129:86'].inputs.noise_seed, 77);
  assert.equal(w['129:131'].inputs.value, true, 'cfg 1 is Fast');
  assert.equal(w['129:86'].inputs.start_at_step, 1);
  assert.deepEqual(w['129:85'].inputs.start_at_step, ['129:125', 0], 'the second sampler is left as shipped');
  assert.equal(w['129:85'].inputs.add_noise, 'disable');
});

test('a low strength skips the high-noise stage and the low-noise sampler adds the noise', () => {
  const w = fresh();
  fillVideoToVideo(w, params({ denoise: 0.25 }), 'clip.mp4', 'wan22-v2v');
  assert.equal(w['129:86'].inputs.start_at_step, 2, 'nothing left for the first sampler to do');
  assert.equal(w['129:85'].inputs.add_noise, 'enable');
  assert.equal(w['129:85'].inputs.start_at_step, 3);
  assert.equal(w['129:85'].inputs.noise_seed, 77);
});

test('High quality and a saved model are applied', () => {
  const w = fresh();
  fillVideoToVideo(w, params({ cfg: 3.5, denoise: 0.8, modelSettings: { files: { highNoiseModel: 'h.safetensors', lowNoiseModel: 'l.safetensors', textEncoder: 'e.safetensors', vae: 'v.safetensors', highNoiseLora: 'hl.safetensors', lowNoiseLora: 'll.safetensors' } } }), 'clip.mp4', 'wan22-v2v');
  assert.equal(w['129:131'].inputs.value, false);
  assert.equal(w['129:86'].inputs.start_at_step, 4);
  assert.equal(w['129:95'].inputs.unet_name, 'h.safetensors', 'the text-to-video model slots apply');
  assert.equal(w['129:90'].inputs.vae_name, 'v.safetensors');
});

test('a source video that is gone is refused when the job is queued', () => {
  assert.throws(() => keepJobSources('wan22-v2v', { sourceVideoPath: '/definitely/not/here.mp4' }, '/sources'), /source video could not be found/);
  assert.deepEqual(keepJobSources('wan22-t2v', { sourceVideoPath: '/anywhere.mp4' }, '/sources'), { sourceVideoPath: '/anywhere.mp4' });
});
