import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FOLDER_LOADERS, MODEL_FOLDERS, MODEL_MANIFEST } from '../shared/modelManifest';
import zImageTemplate from './templates/z-image.json';
import wanTemplate from './templates/wan22-i2v.json';
import wanTextTemplate from './templates/wan22-t2v.json';

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

test('the manifest lists exactly the files the Wan template loads', () => {
  assert.deepEqual(manifestFiles('wan22-i2v'), templateFiles(wanTemplate as Template));
});

test('the manifest lists exactly the files the Wan text-to-video template loads', () => {
  assert.deepEqual(manifestFiles('wan22-t2v'), templateFiles(wanTextTemplate as Template));
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
