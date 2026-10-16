import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ChosenTraits, ModelTraits, TensorShapes, checkFit, readTraits, slotRole } from './modelTraits';

// Headers with the tensor names and shapes read from real files on 2026-10-16 (see the note at the top of modelTraits.ts), cut down
// to the few tensors the rules look at.
const zModel: TensorShapes = { 'cap_embedder.0.weight': [2560], 'cap_embedder.1.weight': [3840, 2560], 'x_embedder.weight': [3840, 64], 'layers.0.attention.qkv.weight': [11520, 3840] };
const qwen3 = (width: number): TensorShapes => ({ 'model.embed_tokens.weight': [151936, width], 'model.layers.0.self_attn.q_proj.weight': [4096, width], 'model.layers.0.self_attn.q_norm.weight': [128], 'model.norm.weight': [width] });
const picVae = (channels: number): TensorShapes => ({ 'encoder.conv_in.weight': [128, 3, 3, 3], 'decoder.conv_in.weight': [512, channels, 3, 3] });
const wanModel = (inChannels: number): TensorShapes => ({ 'patch_embedding.weight': [5120, inChannels, 1, 2, 2], 'text_embedding.0.weight': [5120, 4096] });
const umt5: TensorShapes = { 'shared.weight': [256384, 4096], 'encoder.block.0.layer.0.SelfAttention.q.weight': [4096, 4096], 'encoder.final_layer_norm.weight': [4096] };
const t5: TensorShapes = { ...umt5, 'shared.weight': [32128, 4096] };
const qwen25vl: TensorShapes = { 'model.embed_tokens.weight': [152064, 3584], 'model.layers.0.self_attn.q_proj.weight': [3584, 3584], 'model.layers.0.self_attn.q_proj.bias': [3584], 'model.norm.weight': [3584], 'visual.blocks.0.attn.proj.bias': [1280] };
const llama: TensorShapes = { 'model.embed_tokens.weight': [128256, 4096], 'model.layers.0.self_attn.q_proj.weight': [4096, 4096], 'model.norm.weight': [4096] };
const ovisEncoder: TensorShapes = { ...qwen3(2048), 'vision_model.embeddings.patch_embedding.bias': [1152] };
const gemma: TensorShapes = { ...qwen3(3840), 'vision_model.embeddings.patch_embedding.bias': [1152], 'model.layers.0.pre_feedforward_layernorm.weight': [3840] };
const clipL: TensorShapes = { 'text_model.embeddings.token_embedding.weight': [49408, 768], 'text_model.embeddings.position_embedding.weight': [77, 768] };
const clipG: TensorShapes = { 'text_model.embeddings.token_embedding.weight': [49408, 1280], 'text_projection.weight': [1280, 1280] };
// The "img_in / txt_in" models: Flux (guidance embedder, 4096-wide T5 text), Ovis-Image (a text norm, 2048-wide text) - both double_blocks / single_blocks.
const fluxModel: TensorShapes = { 'img_in.weight': [3072, 64], 'txt_in.weight': [3072, 4096], 'guidance_in.in_layer.bias': [3072], 'double_blocks.0.img_attn.proj.bias': [3072], 'single_blocks.0.linear1.bias': [21504] };
const ovisModel: TensorShapes = { 'img_in.weight': [3072, 64], 'txt_in.weight': [3072, 2048], 'txt_norm.scale': [2048], 'double_blocks.0.img_attn.proj.bias': [3072], 'single_blocks.0.linear1.bias': [33792] };
const hiDreamModel: TensorShapes = { 'x_embedder.proj.bias': [2560], 'caption_projection.0.linear.weight': [2560, 4096], 'double_stream_blocks.0.block.adaLN_modulation.1.bias': [30720], 'final_layer.linear.weight': [64, 2560] };
const aeVae: TensorShapes = { 'decoder.conv_in.weight': [512, 16, 3, 3] };
const flux2Vae: TensorShapes = { 'decoder.conv_in.weight': [512, 32, 3, 3] };
// Qwen-Image (diffusers layout): a different model from Z-Image. Its VAE is Wan 2.1's, a video-style one.
const qwenImageModel: TensorShapes = { 'img_in.weight': [3072, 64], 'txt_in.weight': [3072, 3584], 'txt_norm.weight': [3584], 'transformer_blocks.0.attn.add_q_proj.weight': [3072, 3072] };
const wanVae: TensorShapes = { 'encoder.conv1.weight': [96, 3, 3, 3, 3], 'decoder.conv1.weight': [384, 16, 3, 3, 3], 'conv2.weight': [16, 16, 1, 1, 1] };

test('a Z-Image model reports the text width it takes and the latent channels it works in', () => {
  assert.deepEqual(readTraits(zModel), { arch: 'z-image-dit', textWidth: 2560, latentChannels: 16 });
  // a prefix on every name makes no difference
  const prefixed = Object.fromEntries(Object.entries(zModel).map(([k, v]) => [`model.diffusion_model.${k}`, v]));
  assert.equal(readTraits(prefixed).textWidth, 2560);
});

test('text encoders report the width they give out; Qwen3 has q/k norms, Qwen2.5-VL has q biases, Llama has neither', () => {
  assert.deepEqual(readTraits(qwen3(2560)), { arch: 'qwen3-encoder', outputWidth: 2560 });
  assert.deepEqual(readTraits(qwen25vl), { arch: 'qwen25vl-encoder', outputWidth: 3584 });
  assert.deepEqual(readTraits(llama), { arch: 'llama-encoder', outputWidth: 4096 });
  assert.deepEqual(readTraits(umt5), { arch: 'umt5-encoder', outputWidth: 4096 });
  assert.deepEqual(readTraits(t5), { arch: 't5-encoder', outputWidth: 4096 });
  assert.deepEqual(readTraits(clipL), { arch: 'clip-encoder', outputWidth: 768 });
  assert.deepEqual(readTraits(clipG), { arch: 'clip-encoder', outputWidth: 1280 });
});

test('Ovis 2.5 is a Qwen3 with a vision tower, and Gemma (which has one too) is neither', () => {
  assert.deepEqual(readTraits(ovisEncoder), { arch: 'ovis-encoder', outputWidth: 2048 });
  assert.deepEqual(readTraits(gemma), { arch: 'unknown' });
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

test('a file nothing recognises is unrecognised (hidden, a warning); a file that could not be read is unknown (left on offer)', () => {
  assert.deepEqual(readTraits({ 'something.weight': [1, 2] }), { arch: 'unknown' });
  assert.deepEqual(readTraits({}), { arch: 'unknown' });
  assert.equal(checkFit('z-image', 'vae', { arch: 'unknown' }, {}).status, 'unrecognised');
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

test('Flux, Ovis-Image and HiDream models are recognised and told apart', () => {
  assert.deepEqual(readTraits(fluxModel), { arch: 'flux-dit', textWidth: 4096, latentChannels: 16 });
  assert.deepEqual(readTraits(ovisModel), { arch: 'ovis-dit', textWidth: 2048, latentChannels: 16 });
  assert.deepEqual(readTraits(hiDreamModel), { arch: 'hidream-dit', textWidth: 4096, latentChannels: 16 });
  // a prefix on every name makes no difference
  assert.equal(readTraits(Object.fromEntries(Object.entries(qwenImageModel).map(([k, v]) => [`model.diffusion_model.${k}`, v]))).arch, 'qwen-image-dit');
});

test('each new family accepts its own files and refuses the others', () => {
  const diffusion = { 'qwen-image': qwenImageModel, 'flux-dev': fluxModel, hidream: hiDreamModel, 'ovis-image': ovisModel } as const;
  for (const [family, own] of Object.entries(diffusion)) {
    assert.equal(checkFit(family, 'diffusionModel', t(own), {}).status, 'fits', `${family} takes its own model`);
    for (const [other, shapes] of Object.entries(diffusion)) {
      if (other !== family) assert.equal(checkFit(family, 'diffusionModel', t(shapes), {}).status, 'conflict', `${family} refuses ${other}'s model`);
    }
    assert.equal(checkFit(family, 'diffusionModel', t(zModel), {}).status, 'conflict', `${family} refuses Z-Image's model`);
  }
});

test('Qwen-Image: the Qwen2.5-VL encoder, and a 3D (Qwen / Wan) VAE', () => {
  const chosen: ChosenTraits = { diffusion: [t(qwenImageModel)] };
  assert.equal(checkFit('qwen-image', 'textEncoder', t(qwen25vl), chosen).status, 'fits');
  // the encoders Z-Image and ACE-Step use are the wrong kind or size
  assert.equal(checkFit('qwen-image', 'textEncoder', t(qwen3(2560)), chosen).status, 'conflict');
  assert.equal(checkFit('qwen-image', 'textEncoder', t(qwen3(1024)), chosen).status, 'conflict');
  assert.equal(checkFit('qwen-image', 'vae', t(wanVae), chosen).status, 'fits');
  assert.equal(checkFit('qwen-image', 'vae', t(aeVae), chosen).status, 'conflict');
});

test('Flux: T5 and CLIP-L each in their own slot, the Flux VAE, and no 32-channel Flux 2 VAE', () => {
  const chosen: ChosenTraits = { diffusion: [t(fluxModel)] };
  assert.equal(checkFit('flux-dev', 'textEncoder', t(t5), chosen).status, 'fits');
  assert.equal(checkFit('flux-dev', 'clipL', t(clipL), chosen).status, 'fits');
  // swapped, or the Z-Image encoder (the case that looked plausible in the dropdown)
  assert.equal(checkFit('flux-dev', 'textEncoder', t(clipL), chosen).status, 'conflict');
  assert.equal(checkFit('flux-dev', 'clipL', t(t5), chosen).status, 'conflict');
  assert.equal(checkFit('flux-dev', 'textEncoder', t(qwen3(2560)), chosen).status, 'conflict');
  assert.equal(checkFit('flux-dev', 'clipL', t(clipG), chosen).status, 'conflict');
  assert.equal(checkFit('flux-dev', 'vae', t(aeVae), chosen).status, 'fits');
  assert.equal(checkFit('flux-dev', 'vae', t(flux2Vae), chosen).status, 'conflict');
  assert.equal(checkFit('flux-dev', 'vae', t(wanVae), chosen).status, 'conflict');
});

test('HiDream: four encoders, each in its own slot', () => {
  const chosen: ChosenTraits = { diffusion: [t(hiDreamModel)], encoder: [t(llama)] };
  assert.equal(checkFit('hidream', 'textEncoder', t(t5), chosen).status, 'fits');
  assert.equal(checkFit('hidream', 'llama', t(llama), { diffusion: [t(hiDreamModel)], encoder: [t(t5)] }).status, 'fits');
  assert.equal(checkFit('hidream', 'clipL', t(clipL), chosen).status, 'fits');
  assert.equal(checkFit('hidream', 'clipG', t(clipG), chosen).status, 'fits');
  assert.equal(checkFit('hidream', 'clipG', t(clipL), chosen).status, 'conflict');
  assert.equal(checkFit('hidream', 'llama', t(qwen25vl), chosen).status, 'conflict');
  assert.equal(checkFit('hidream', 'textEncoder', t(umt5), chosen).status, 'conflict');
  assert.equal(checkFit('hidream', 'vae', t(aeVae), chosen).status, 'fits');
});

test('Ovis-Image: the Ovis 2.5 encoder, not a plain Qwen3 of the same width', () => {
  const chosen: ChosenTraits = { diffusion: [t(ovisModel)] };
  assert.equal(checkFit('ovis-image', 'textEncoder', t(ovisEncoder), chosen).status, 'fits');
  assert.equal(checkFit('ovis-image', 'textEncoder', t(qwen3(2048)), chosen).status, 'conflict');
  assert.equal(checkFit('ovis-image', 'textEncoder', t(qwen3(2560)), chosen).status, 'conflict');
  // and Z-Image does not take Ovis's
  assert.equal(checkFit('z-image', 'textEncoder', t(ovisEncoder), {}).status, 'conflict');
});

test('the CLIP and Llama slots are checked, and a Wan encoder slot is still the T5 kind, not plain T5', () => {
  assert.equal(slotRole('clipL'), 'auxEncoder');
  assert.equal(slotRole('llama'), 'encoder');
  assert.equal(checkFit('wan22-i2v', 'textEncoder', t(t5), {}).status, 'conflict');
});

test('the picked image model decides which encoders and VAEs fit, and the reverse', () => {
  const chosen: ChosenTraits = { diffusion: [t(zModel)] };
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(2560)), chosen).status, 'fits');
  const wide = checkFit('z-image', 'textEncoder', t(qwen3(4096)), chosen);
  assert.equal(wide.status, 'conflict');
  assert.match((wide as { reason: string }).reason, /gives 4096-wide text.*needs 2560/);
  assert.equal(checkFit('z-image', 'vae', t(picVae(16)), chosen).status, 'fits');
  assert.equal(checkFit('z-image', 'vae', t(picVae(4)), chosen).status, 'conflict');
  // the other way round: a picked 4-channel VAE rules out the 16-channel model
  const back = checkFit('z-image', 'diffusionModel', t(zModel), { vae: [t(picVae(4))] });
  assert.equal(back.status, 'conflict');
  assert.match((back as { reason: string }).reason, /16 latent channels.*uses 4/);
});

test('an unrecognised pick constrains nothing', () => {
  const chosen: ChosenTraits = { diffusion: [{ arch: 'unknown' }] };
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(2560)), chosen).status, 'fits');
});

test('the family decides what fits even when nothing is picked: a small Qwen3 encoder is the wrong size for Z-Image', () => {
  // e.g. the 0.6B Qwen3 that ACE-Step uses: the right kind of encoder, the wrong size
  const small = checkFit('z-image', 'textEncoder', t(qwen3(1024)), {});
  assert.equal(small.status, 'conflict');
  assert.match((small as { reason: string }).reason, /1024-wide.*needs 2560/);
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(2560)), {}).status, 'fits');
  assert.equal(checkFit('z-image', 'vae', t(picVae(4)), {}).status, 'conflict');
});

test('a Qwen-Image model is recognised, so it is told apart from a Z-Image one', () => {
  assert.deepEqual(readTraits(qwenImageModel), { arch: 'qwen-image-dit', textWidth: 3584, latentChannels: 16 });
  const fit = checkFit('z-image', 'diffusionModel', t(qwenImageModel), {});
  assert.equal(fit.status, 'conflict');
  assert.match((fit as { reason: string }).reason, /Qwen-Image model, not a Z-Image model/);
  assert.equal(checkFit('wan22-i2v', 'highNoiseModel', t(qwenImageModel), {}).status, 'conflict');
});

test('the built-in VAE entries in ComfyUI are not VAE files', () => {
  assert.equal(checkFit('z-image', 'vae', null, {}, 'pixel_space').status, 'conflict');
  assert.equal(checkFit('z-image', 'vae', null, {}, 'taesd').status, 'conflict');
  assert.equal(checkFit('z-image', 'vae', null, {}, 'ae.safetensors').status, 'unknown');
  // only a VAE slot is asked
  assert.equal(checkFit('z-image', 'textEncoder', null, {}, 'pixel_space').status, 'unknown');
});

test('what is already picked still narrows the rest (a Z-Image pick next to a wrong-size encoder)', () => {
  assert.equal(checkFit('z-image', 'textEncoder', t(qwen3(2560)), { diffusion: [t(zModel)] }).status, 'fits');
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
