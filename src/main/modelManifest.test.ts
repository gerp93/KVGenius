import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FOLDER_LOADERS, MODEL_FOLDERS, MODEL_MANIFEST } from '../shared/modelManifest';
import zImageTemplate from './templates/z-image.json';
import qwenTemplate from './templates/qwen-image.json';
import fluxTemplate from './templates/flux-dev.json';
import hiDreamTemplate from './templates/hidream.json';
import ovisTemplate from './templates/ovis-image.json';
import wanTemplate from './templates/wan22-i2v.json';
import wanTextTemplate from './templates/wan22-t2v.json';
import faceKleinTemplate from './templates/face-klein.json';
import faceQwenTemplate from './templates/face-qwen.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;

/** Every file name a template's loader nodes ask for, as "folder/name". */
function templateFiles(template: Template): string[] {
  const found: string[] = [];
  for (const node of Object.values(template)) {
    for (const folder of MODEL_FOLDERS) {
      const loader = FOLDER_LOADERS[folder];
      const value = node.inputs[loader.input];
      // Upscale templates carry a placeholder the user's choice replaces; only real file names count.
      if (node.class_type === loader.node && typeof value === 'string' && !value.startsWith('__')) found.push(`${folder}/${value}`);
    }
  }
  return found.sort();
}

function manifestFiles(family: string): string[] {
  return MODEL_MANIFEST.filter((f) => f.family === family)
    .flatMap((f) => f.files.map((file) => `${file.folder}/${file.file}`))
    .sort();
}

test('the manifest lists exactly the files the Z Image template loads', () => {
  assert.deepEqual(manifestFiles('z-image'), templateFiles(zImageTemplate as Template));
});

/** Like templateFiles, but also the text encoders loaded by the multi-file loaders (Dual / Quadruple CLIP loaders). */
function allTemplateFiles(template: Template): string[] {
  const found = templateFiles(template);
  for (const node of Object.values(template)) {
    if (node.class_type !== 'DualCLIPLoader' && node.class_type !== 'QuadrupleCLIPLoader') continue;
    for (const [key, value] of Object.entries(node.inputs)) if (/^clip_name\d$/.test(key) && typeof value === 'string') found.push(`text_encoders/${value}`);
  }
  return found.sort();
}

for (const [family, template] of [
  ['qwen-image', qwenTemplate],
  ['flux-dev', fluxTemplate],
  ['hidream', hiDreamTemplate],
  ['ovis-image', ovisTemplate],
] as const) {
  test(`the manifest lists exactly the files the ${family} template loads`, () => {
    assert.deepEqual(manifestFiles(family), allTemplateFiles(template as Template));
  });
}

test('the manifest lists exactly the files the Wan template loads', () => {
  assert.deepEqual(manifestFiles('wan22-i2v'), templateFiles(wanTemplate as Template));
});

test('the manifest lists exactly the files the Wan text-to-video template loads', () => {
  assert.deepEqual(manifestFiles('wan22-t2v'), templateFiles(wanTextTemplate as Template));
});

test('the manifest lists exactly the files the face-reference templates load', () => {
  assert.deepEqual(manifestFiles('face-klein'), templateFiles(faceKleinTemplate as Template));
  assert.deepEqual(manifestFiles('face-qwen'), templateFiles(faceQwenTemplate as Template));
});

test('manifest ids are unique and no feature lists a file twice', () => {
  const ids = MODEL_MANIFEST.map((f) => f.id);
  assert.equal(new Set(ids).size, ids.length);
  // Features may share a file (text and image to video use one text encoder and VAE); the download planner drops repeats.
  for (const feature of MODEL_MANIFEST) {
    const own = feature.files.map((x) => `${x.folder}/${x.file}`);
    assert.equal(new Set(own).size, own.length, feature.id);
  }
});
