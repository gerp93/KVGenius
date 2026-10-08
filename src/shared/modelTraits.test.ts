import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ChosenTraits, ModelTraits, TensorShapes, checkFit, readTraits, slotRole } from './modelTraits';

// Headers built from the rules' own assumptions about the tensor names (see the note at the top of modelTraits.ts): these check the
// logic, and cannot tell whether a real file names its tensors this way.
const zModel: TensorShapes = { 'cap_embedder.0.weight': [2560], 'cap_embedder.1.weight': [3840, 2560], 'x_embedder.weight': [3840, 64], 'layers.0.attention.qkv.weight': [11520, 3840] };
const qwen3 = (width: number): TensorShapes => ({ 'model.embed_tokens.weight': [151936, width], 'model.layers.0.self_attn.q_norm.weight': [128], 'model.norm.weight': [width] });
const picVae = (channels: number): TensorShapes => ({ 'encoder.conv_in.weight': [128, 3, 3, 3], 'decoder.conv_in.weight': [512, channels, 3, 3] });
const wanModel = (inChannels: number): TensorShapes => ({ 'patch_embedding.weight': [5120, inChannels, 1, 2, 2], 'text_embedding.0.weight': [5120, 4096] });
const umt5: TensorShapes = { 'shared.weight': [256384, 4096], 'encoder.block.0.layer.0.SelfAttention.q.weight': [4096, 4096], 'encoder.final_layer_norm.weight': [4096] };
const wanVae: TensorShapes = { 'encoder.conv1.weight': [96, 3, 3, 3, 3], 'decoder.conv1.weight': [384, 16, 3, 3, 3], 'conv2.weight': [16, 16, 1, 1, 1] };

test('a Z-Image model reports the text width it takes and the latent channels it works in', () => {
  assert.deepEqual(readTraits(zModel), { arch: 'z-image-dit', textWidth: 2560, latentChannels: 16 });
  // a prefix on every name makes no difference
  const prefixed = Object.fromEntries(Object.entries(zModel).map(([k, v]) => [`model.diffusion_model.${k}`, v]));
  assert.equal(readTraits(prefixed).textWidth, 2560);
});

test('text encoders report the width they give out; Qwen3 is told from Qwen2 by its q/k norms', () => {
  assert.deepEqual(readTraits(qwen3(2560)), { arch: 'qwen3-encoder', outputWidth: 2560 });
  const qwen2 = qwen3(2560);
  delete qwen2['model.layers.0.self_attn.q_norm.weight'];
  assert.equal(readTraits(qwen2).arch, 'unknown');
  assert.deepEqual(readTraits(umt5), { arch: 'umt5-encoder', outputWidth: 4096 });
});

test('VAEs report their latent channels, and a video VAE is told from a picture VAE', () => {
  assert.deepEqual(readTraits(picVae(16)), { arch: 'latent-vae', latentChannels: 16, vaeDims: 2 });
  assert.deepEqual(readTraits(picVae(4)), { arch: 'latent-vae', latentChannels: 4, vaeDims: 2 });
  assert.deepEqual(readTraits(wanVae), { arch: 'latent-vae', latentChannels: 16, vaeDims: 3 });
});

test('Wan models are told apart by their input channels', () => {
  assert.equal(readTraits(wanModel(16)).variant, 't2v');
  assert.equal(readTraits(wanModel(36)).variant, 'i2v');
  assert.equal(readTraits(wanModel(48)).variant, undefined);
  assert.equal(readTraits(wanModel(36)).textWidth, 4096);
});

test('a file nothing recognises is unknown, and an unknown file is never in conflict', () => {
  assert.deepEqual(readTraits({ 'something.weight': [1, 2] }), { arch: 'unknown' });
  assert.deepEqual(readTraits({}), { arch: 'unknown' });
  assert.equal(checkFit('z-image', 'vae', { arch: 'unknown' }, {}).status, 'unknown');
  assert.equal(checkFit('z-image', 'vae', null, {}).status, 'unknown');
  assert.equal(checkFit('z-image', 'unknownSlot', readTraits(picVae(16)), {}).status, 'unknown');
});

const t = (shapes: TensorShapes): ModelTraits => readTraits(shapes);

test('with nothing picked, a file only has to be the right kind for the slot', () => {
  assert.equal(checkFit('z-image', 'diffusionModel', t(zModel), {}).status, 'fits');
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(2560)), {}).status, 'fits');
  assert.equal(checkFit('z-image', 'vae', t(picVae(16)), {}).status, 'fits');
  const wrong = checkFit('z-image', 'diffusionModel', t(wanModel(16)), {});
  assert.equal(wrong.status, 'conflict');
  assert.match((wrong as { reason: string }).reason, /Wan video model/);
  assert.equal(checkFit('z-image', 'textEncoder', t(umt5), {}).status, 'conflict');
});

test('the picked image model decides which encoders and VAEs fit, and the reverse', () => {
  const chosen: ChosenTraits = { diffusion: [t(zModel)] };
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(2560)), chosen).status, 'fits');
  const wide = checkFit('z-image', 'textEncoder', t(qwen3(4096)), chosen);
  assert.equal(wide.status, 'conflict');
  assert.match((wide as { reason: string }).reason, /gives 4096-wide text.*takes 2560/);
  assert.equal(checkFit('z-image', 'vae', t(picVae(16)), chosen).status, 'fits');
  assert.equal(checkFit('z-image', 'vae', t(picVae(4)), chosen).status, 'conflict');
  // the other way round: a picked 4-channel VAE rules out the 16-channel model
  const back = checkFit('z-image', 'diffusionModel', t(zModel), { vae: [t(picVae(4))] });
  assert.equal(back.status, 'conflict');
  assert.match((back as { reason: string }).reason, /16 latent channels.*uses 4/);
});

test('an unrecognised pick constrains nothing', () => {
  const chosen: ChosenTraits = { diffusion: [{ arch: 'unknown' }] };
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(4096)), chosen).status, 'fits');
});

test('Wan: text to video and image to video models do not swap, nor do picture and video VAEs', () => {
  assert.equal(checkFit('wan22-i2v', 'highNoiseModel', t(wanModel(36)), {}).status, 'fits');
  const swapped = checkFit('wan22-i2v', 'lowNoiseModel', t(wanModel(16)), {});
  assert.equal(swapped.status, 'conflict');
  assert.match((swapped as { reason: string }).reason, /text-to-video model/);
  assert.equal(checkFit('wan22-t2v', 'highNoiseModel', t(wanModel(16)), {}).status, 'fits');
  assert.equal(checkFit('wan22-t2v', 'highNoiseModel', t(wanModel(36)), {}).status, 'conflict');
  assert.equal(checkFit('wan22-t2v', 'vae', t(wanVae), {}).status, 'fits');
  assert.equal(checkFit('wan22-t2v', 'vae', t(picVae(16)), {}).status, 'conflict');
  assert.equal(checkFit('z-image', 'vae', t(wanVae), {}).status, 'conflict');
});

test('Wan: both diffusion slots count when checking the encoder', () => {
  const chosen: ChosenTraits = { diffusion: [t(wanModel(36)), t(wanModel(36))] };
  assert.equal(checkFit('wan22-i2v', 'textEncoder', t(umt5), chosen).status, 'fits');
  assert.equal(checkFit('wan22-i2v', 'textEncoder', t(qwen3(2560)), chosen).status, 'conflict');
});

test('LoRA slots are not checked', () => {
  assert.equal(slotRole('highNoiseLora'), null);
  assert.equal(checkFit('wan22-i2v', 'highNoiseLora', t(wanModel(36)), {}).status, 'unknown');
});
