import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MODEL_MANIFEST } from '../shared/modelManifest';
import { DEFAULT_DENOISE } from '../shared/imageToImage';
import { fillImageToImage, Z_IMAGE_I2I_NODE_MAP } from './imageToImagePatch';
import i2iTemplate from './templates/z-image-i2i.json';
import textTemplate from './templates/z-image.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const fresh = () => JSON.parse(JSON.stringify(i2iTemplate)) as Template;
const params = { prompt: 'a fox', width: 832, height: 1216, seed: 7, steps: 8, cfg: 1 };

test('the template is the text-to-image graph with the empty latent replaced by load -> scale -> encode', () => {
  const t = fresh();
  const text = textTemplate as unknown as Template;
  assert.equal(t['57:13'], undefined, 'no empty latent any more');
  assert.equal(t[Z_IMAGE_I2I_NODE_MAP.loadImage].class_type, 'LoadImage');
  assert.equal(t[Z_IMAGE_I2I_NODE_MAP.scale].class_type, 'ImageScale');
  assert.equal(t[Z_IMAGE_I2I_NODE_MAP.encode].class_type, 'VAEEncode');
  assert.deepEqual(t[Z_IMAGE_I2I_NODE_MAP.scale].inputs.image, [Z_IMAGE_I2I_NODE_MAP.loadImage, 0]);
  assert.deepEqual(t[Z_IMAGE_I2I_NODE_MAP.encode].inputs.pixels, [Z_IMAGE_I2I_NODE_MAP.scale, 0]);
  assert.deepEqual(t[Z_IMAGE_I2I_NODE_MAP.encode].inputs.vae, ['57:29', 0], 'encodes with the same VAE that decodes');
  assert.deepEqual(t[Z_IMAGE_I2I_NODE_MAP.sampler].inputs.latent_image, [Z_IMAGE_I2I_NODE_MAP.encode, 0]);
  // every other node is exactly the text-to-image one, so profiles and the manifest still describe it
  for (const id of Object.keys(text)) {
    if (id === '57:13' || id === '9') continue;
    const same = id === Z_IMAGE_I2I_NODE_MAP.sampler ? { ...text[id].inputs, latent_image: [Z_IMAGE_I2I_NODE_MAP.encode, 0] } : text[id].inputs;
    assert.deepEqual(t[id].inputs, same, `node ${id}`);
    assert.equal(t[id].class_type, text[id].class_type);
  }
  // nothing is left pointing at a node that does not exist
  for (const [id, node] of Object.entries(t)) {
    for (const value of Object.values(node.inputs)) {
      if (Array.isArray(value) && typeof value[0] === 'string') assert.ok(t[value[0]], `${id} refers to ${value[0]}`);
    }
  }
});

test('the manifest files for Z-Image are exactly what the image-to-image template loads', () => {
  const loaded = Object.values(fresh())
    .flatMap((n) => [n.inputs.unet_name, n.inputs.clip_name, n.inputs.vae_name])
    .filter((v): v is string => typeof v === 'string')
    .sort();
  const listed = MODEL_MANIFEST.find((f) => f.id === 'z-image')!.files.map((f) => f.file).sort();
  assert.deepEqual(loaded, listed);
});

test('filling it sets the picture, size, prompt, sampler values and the strength', () => {
  const t = fresh();
  fillImageToImage(t, { ...params, denoise: 0.45 }, 'uploaded.png');
  assert.equal(t['i2i-load'].inputs.image, 'uploaded.png');
  assert.equal(t['i2i-scale'].inputs.width, 832);
  assert.equal(t['i2i-scale'].inputs.height, 1216);
  assert.equal(t['57:27'].inputs.text, 'a fox');
  assert.deepEqual([t['57:3'].inputs.seed, t['57:3'].inputs.steps, t['57:3'].inputs.cfg, t['57:3'].inputs.denoise], [7, 8, 1, 0.45]);
});

test('without a strength the default is used, and a model profile swaps files and sampler as in text to image', () => {
  const t = fresh();
  fillImageToImage(
    t,
    {
      ...params,
      modelName: 'Photoreal',
      modelSettings: { files: { diffusionModel: 'photoreal.safetensors', textEncoder: 'enc.safetensors', vae: 'v2.safetensors' }, sampler: 'euler', scheduler: 'karras', shift: 5 },
    },
    'u.png',
  );
  assert.equal(t['57:3'].inputs.denoise, DEFAULT_DENOISE);
  assert.equal(t['57:28'].inputs.unet_name, 'photoreal.safetensors');
  assert.equal(t['57:30'].inputs.clip_name, 'enc.safetensors');
  assert.equal(t['57:29'].inputs.vae_name, 'v2.safetensors');
  assert.equal(t['57:3'].inputs.sampler_name, 'euler');
  assert.equal(t['57:11'].inputs.shift, 5);
});

test('a template that drifts from the patch fails loudly', () => {
  const t = fresh();
  delete (t as Record<string, unknown>)['i2i-scale'];
  assert.throws(() => fillImageToImage(t, params, 'u.png'), /no node i2i-scale/);
});
