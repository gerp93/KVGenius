import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MODEL_MANIFEST } from '../shared/modelManifest';
import { DEFAULT_DENOISE } from '../shared/imageToImage';
import { fillImageToImage, fillOutpaint, Z_IMAGE_I2I_NODE_MAP } from './imageToImagePatch';
import outpaintTemplate from './templates/z-image-outpaint.json';
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

import inpaintTemplate from './templates/z-image-inpaint.json';
import { fillInpaint, Z_IMAGE_INPAINT_NODE_MAP } from './imageToImagePatch';

const freshInpaint = () => JSON.parse(JSON.stringify(inpaintTemplate)) as Template;

test('the inpainting template is image to image plus the mask steps, and every reference resolves', () => {
  const t = freshInpaint();
  const i2i = fresh();
  const added = Object.keys(t).filter((id) => !(id in i2i));
  assert.deepEqual(added.sort(), ['ip-composite', 'ip-img2mask', 'ip-maskblur', 'ip-maskload', 'ip-mask2img', 'ip-maskscale', 'ip-setmask'].sort());
  for (const [id, node] of Object.entries(t)) {
    for (const value of Object.values(node.inputs)) {
      if (Array.isArray(value) && typeof value[0] === 'string') assert.ok(t[value[0]], `${id} refers to ${value[0]}`);
    }
  }
  // the sampler works on the masked latent; the saved picture is the result pasted back over the (scaled) original
  assert.deepEqual(t[Z_IMAGE_INPAINT_NODE_MAP.sampler].inputs.latent_image, ['ip-setmask', 0]);
  assert.deepEqual(t['ip-setmask'].inputs.samples, [Z_IMAGE_INPAINT_NODE_MAP.encode, 0]);
  assert.equal(t['ip-composite'].class_type, 'ImageCompositeMasked');
  assert.deepEqual(t['ip-composite'].inputs.destination, [Z_IMAGE_INPAINT_NODE_MAP.scale, 0]);
  assert.deepEqual(t['ip-composite'].inputs.source, ['57:8', 0]);
  assert.deepEqual(t['9'].inputs.images, ['ip-composite', 0]);
  // the same mask limits the sampler and blends the paste, and it is the picture's own fit (centre crop) at the output size
  assert.deepEqual(t['ip-setmask'].inputs.mask, ['ip-img2mask', 0]);
  assert.deepEqual(t['ip-composite'].inputs.mask, ['ip-img2mask', 0]);
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.maskScale].inputs.crop, 'center');
  // every model node is the text-to-image one, so profiles and the manifest still describe it
  const text = textTemplate as unknown as Template;
  for (const id of ['57:30', '57:29', '57:28', '57:27', '57:11']) assert.deepEqual(t[id].inputs, text[id].inputs, `node ${id}`);
});

test('filling the inpainting template sets the picture, the mask, the size and the sampler', () => {
  const t = freshInpaint();
  fillInpaint(t, { ...params, denoise: 0.8 }, 'photo.png', 'mask.png');
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.loadImage].inputs.image, 'photo.png');
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.loadMask].inputs.image, 'mask.png');
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.scale].inputs.width, 832);
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.maskScale].inputs.width, 832);
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.maskScale].inputs.height, 1216);
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.sampler].inputs.denoise, 0.8);
  assert.equal(t[Z_IMAGE_INPAINT_NODE_MAP.prompt].inputs.text, 'a fox');
});

test('a model profile applies to inpainting as to the other Z-Image graphs', () => {
  const t = freshInpaint();
  fillInpaint(
    t,
    { ...params, modelSettings: { files: { diffusionModel: 'other.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' }, sampler: 'euler', scheduler: 'karras', shift: 5 } },
    'p.png',
    'm.png',
  );
  assert.equal(t['57:28'].inputs.unet_name, 'other.safetensors');
});

test('the outpainting template pads the picture and takes its mask from the padding; nothing points at a missing node', () => {
  const t = JSON.parse(JSON.stringify(outpaintTemplate)) as Template;
  assert.equal(t['ip-maskload'], undefined, 'no painted mask file');
  assert.equal(t['op-pad'].class_type, 'ImagePadForOutpaint');
  assert.deepEqual(t['op-pad'].inputs.image, ['i2i-load', 0]);
  assert.deepEqual(t['i2i-scale'].inputs.image, ['op-pad', 0]);
  assert.deepEqual(t['ip-mask2img'].inputs.mask, ['op-pad', 1]);
  assert.equal(t['i2i-scale'].inputs.crop, 'disabled', 'the output is the whole canvas, never cropped');
  // the sampler starts from the picture with a blurred stretch of itself where the new area is, not from grey
  assert.deepEqual(t['op-bg'].inputs.image, ['i2i-load', 0]);
  assert.deepEqual(t['op-bgblur'].inputs.image, ['op-bg', 0]);
  assert.equal(t['op-init'].class_type, 'ImageCompositeMasked');
  assert.deepEqual(t['op-init'].inputs.destination, ['op-bgblur', 0]);
  assert.deepEqual(t['op-init'].inputs.source, ['i2i-scale', 0]);
  assert.deepEqual(t['op-keepmask'].inputs.mask, ['ip-img2mask', 0]);
  assert.deepEqual(t['i2i-encode'].inputs.pixels, ['op-init', 0]);
  assert.deepEqual(t['ip-composite'].inputs.destination, ['op-init', 0]);
  for (const [id, node] of Object.entries(t)) {
    for (const value of Object.values(node.inputs)) {
      if (Array.isArray(value) && typeof value[0] === 'string') assert.ok(t[value[0]], `${id} refers to ${value[0]}`);
    }
  }
});

test('filling the outpainting template sets the padding, the canvas size and a full-strength sampler', () => {
  const t = JSON.parse(JSON.stringify(outpaintTemplate)) as Template;
  fillOutpaint(t, { ...params, width: 1536, height: 1024, denoise: 0.3 }, 'up.png', { left: 512, top: 0, right: 0, bottom: 64 });
  assert.equal(t['i2i-load'].inputs.image, 'up.png');
  assert.deepEqual([t['op-pad'].inputs.left, t['op-pad'].inputs.top, t['op-pad'].inputs.right, t['op-pad'].inputs.bottom], [512, 0, 0, 64]);
  assert.deepEqual([t['i2i-scale'].inputs.width, t['i2i-scale'].inputs.height], [1536, 1024]);
  assert.deepEqual([t['ip-maskscale'].inputs.width, t['ip-maskscale'].inputs.height], [1536, 1024]);
  assert.deepEqual([t['op-bg'].inputs.width, t['op-bg'].inputs.height], [1536, 1024]);
  assert.equal(t['57:3'].inputs.denoise, 0.3, 'the chosen strength reaches the sampler');
  const fresh = JSON.parse(JSON.stringify(outpaintTemplate)) as Template;
  fillOutpaint(fresh, { ...params, width: 1536, height: 1024 }, 'up.png', { left: 512, top: 0, right: 0, bottom: 0 });
  assert.equal(fresh['57:3'].inputs.denoise, 0.8, 'without one, the outpainting default');
  assert.equal(t['57:27'].inputs.text, 'a fox');
});
