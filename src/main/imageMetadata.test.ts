import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as zlib from 'zlib';
import { readPngTextChunks, settingsFromPng } from './imageMetadata';

function crc32(buf: Buffer): number {
  let c = ~0;
  for (const byte of buf) {
    c ^= byte;
    for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0xedb88320 : c >>> 1;
  }
  return ~c >>> 0;
}

function chunk(type: string, data: Buffer): Buffer {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write(type, 4, 'latin1');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0);
  return Buffer.concat([head, data, crc]);
}

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const IHDR = chunk('IHDR', Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]));
const IDAT = chunk('IDAT', zlib.deflateSync(Buffer.from([0, 0, 0, 0])));
const IEND = chunk('IEND', Buffer.alloc(0));

function tEXt(keyword: string, text: string): Buffer {
  return chunk('tEXt', Buffer.concat([Buffer.from(keyword, 'latin1'), Buffer.from([0]), Buffer.from(text, 'utf8')]));
}

function png(...chunks: Buffer[]): Buffer {
  return Buffer.concat([SIGNATURE, IHDR, ...chunks, IDAT, IEND]);
}

const comfyPrompt = {
  '57:28': { class_type: 'UNETLoader', inputs: { unet_name: 'photoreal.safetensors', weight_dtype: 'default' } },
  '57:30': { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen_3_4b.safetensors' } },
  '57:29': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
  '57:11': { class_type: 'ModelSamplingAuraFlow', inputs: { shift: 3, model: ['57:28', 0] } },
  '57:3': { class_type: 'KSampler', inputs: { seed: 5, steps: 30, cfg: 4.5, sampler_name: 'euler', scheduler: 'karras', denoise: 1, model: ['57:11', 0] } },
};

test('ComfyUI settings are read from the workflow a picture carries', () => {
  const settings = settingsFromPng(png(tEXt('prompt', JSON.stringify(comfyPrompt))));
  assert.deepEqual(settings, {
    source: 'comfyui',
    steps: 30,
    cfg: 4.5,
    sampler: 'euler',
    scheduler: 'karras',
    shift: 3,
    fileHints: { diffusionModel: 'photoreal.safetensors', textEncoder: 'qwen_3_4b.safetensors', vae: 'ae.safetensors' },
  });
});

test('compressed and international text chunks are read too', () => {
  const body = Buffer.from(JSON.stringify(comfyPrompt), 'utf8');
  const zTXt = chunk('zTXt', Buffer.concat([Buffer.from('prompt\0'), Buffer.from([0]), zlib.deflateSync(body)]));
  assert.equal(settingsFromPng(png(zTXt))?.steps, 30);
  const plainITXt = chunk('iTXt', Buffer.concat([Buffer.from('prompt\0'), Buffer.from([0, 0]), Buffer.from('\0\0'), body]));
  assert.equal(settingsFromPng(png(plainITXt))?.cfg, 4.5);
  const packedITXt = chunk('iTXt', Buffer.concat([Buffer.from('prompt\0'), Buffer.from([1, 0]), Buffer.from('\0\0'), zlib.deflateSync(body)]));
  assert.equal(settingsFromPng(png(packedITXt))?.scheduler, 'karras');
});

test('an Automatic1111 parameters text gives steps and CFG only', () => {
  const settings = settingsFromPng(png(tEXt('parameters', 'a fox\nNegative prompt: blurry\nSteps: 28, Sampler: DPM++ 2M Karras, CFG scale: 6.5, Seed: 1')));
  assert.deepEqual(settings, { source: 'a1111', steps: 28, cfg: 6.5 });
});

test('pictures with nothing usable, damaged data or other file types give nothing', () => {
  assert.equal(settingsFromPng(png()), null);
  assert.equal(settingsFromPng(png(tEXt('prompt', 'not json'))), null);
  assert.equal(settingsFromPng(png(tEXt('prompt', JSON.stringify({ '1': { class_type: 'SaveImage', inputs: {} } })))), null);
  assert.equal(settingsFromPng(png(tEXt('parameters', 'just a caption'))), null);
  assert.equal(settingsFromPng(Buffer.from('GIF89a not a png')), null);
  assert.equal(settingsFromPng(Buffer.alloc(0)), null);
  const truncated = png(tEXt('prompt', JSON.stringify(comfyPrompt))).subarray(0, 40);
  assert.deepEqual(readPngTextChunks(truncated), {});
});

test('a sampler whose values come from other nodes (links) is not read as a number', () => {
  const linked = { '1': { class_type: 'KSampler', inputs: { steps: ['5', 0], cfg: ['6', 0], sampler_name: 'euler', scheduler: 'simple' } } };
  assert.equal(settingsFromPng(png(tEXt('prompt', JSON.stringify(linked)))), null);
});
