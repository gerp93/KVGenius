import { test } from 'node:test';
import assert from 'node:assert/strict';
import zImage from './templates/z-image.json';
import i2i from './templates/z-image-i2i.json';
import inpaint from './templates/z-image-inpaint.json';
import outpaint from './templates/z-image-outpaint.json';
import i2v from './templates/wan22-i2v.json';
import t2v from './templates/wan22-t2v.json';
import v2v from './templates/wan22-v2v.json';
import upscaleImage from './templates/upscale-image.json';
import upscaleVideo from './templates/upscale-video.json';

type Template = Record<string, { class_type: string; inputs: Record<string, unknown> }>;
const templates: Record<string, Template> = {
  'z-image': zImage as unknown as Template,
  'z-image-i2i': i2i as unknown as Template,
  'z-image-inpaint': inpaint as unknown as Template,
  'z-image-outpaint': outpaint as unknown as Template,
  'wan22-i2v': i2v as unknown as Template,
  'wan22-t2v': t2v as unknown as Template,
  'wan22-v2v': v2v as unknown as Template,
  'upscale-image': upscaleImage as unknown as Template,
  'upscale-video': upscaleVideo as unknown as Template,
};

/** Ranges ComfyUI itself enforces on a node's literal inputs (a value outside them fails the whole prompt with a 400 before anything runs). */
const RANGES: Record<string, Record<string, { min: number; max: number }>> = {
  ImageBlur: { blur_radius: { min: 1, max: 31 }, sigma: { min: 0.1, max: 10 } },
  ImageScale: { width: { min: 0, max: 16384 }, height: { min: 0, max: 16384 } },
  ImageFromBatch: { batch_index: { min: 0, max: 4095 }, length: { min: 1, max: 4096 } },
};

test('every literal a template sets on a node stays inside the range ComfyUI accepts for it', () => {
  let checked = 0;
  for (const [name, template] of Object.entries(templates)) {
    for (const [id, node] of Object.entries(template)) {
      const ranges = RANGES[node.class_type];
      if (!ranges) continue;
      for (const [input, range] of Object.entries(ranges)) {
        const value = node.inputs[input];
        if (typeof value !== 'number') continue; // a link, or a placeholder the patch fills in
        checked++;
        assert.ok(value >= range.min && value <= range.max, `${name} ${id} (${node.class_type}) ${input} = ${value}, must be ${range.min}-${range.max}`);
      }
    }
  }
  assert.ok(checked > 0, 'something was checked');
});
