import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cleanDeviceName, formatVram, gpuLabel, parseSystemStats } from './gpuInfo';

const GIB = 1024 ** 3;

test('a device name loses its torch prefix and allocator suffix', () => {
  assert.equal(cleanDeviceName('cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync'), 'NVIDIA GeForce RTX 4090');
  assert.equal(cleanDeviceName('NVIDIA GeForce RTX 3060'), 'NVIDIA GeForce RTX 3060');
});

test('system stats are read leniently', () => {
  const stats = {
    devices: [
      { name: 'cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync', type: 'cuda', vram_total: 24 * GIB, vram_free: 20 * GIB },
      { type: 'cuda' },
      null,
    ],
  };
  const gpus = parseSystemStats(stats);
  assert.equal(gpus.length, 1);
  assert.equal(gpus[0].name, 'NVIDIA GeForce RTX 4090');
  assert.equal(gpus[0].vramFree, 20 * GIB);
  assert.deepEqual(parseSystemStats(null), []);
  assert.deepEqual(parseSystemStats({ devices: 'x' }), []);
});

test('the sidebar label is short', () => {
  const gpu = { name: 'NVIDIA GeForce RTX 4090', type: 'cuda', vramTotal: 24 * GIB, vramFree: 0 };
  assert.equal(gpuLabel(gpu), 'RTX 4090 · 24 GB');
  assert.equal(gpuLabel({ ...gpu, name: 'AMD Radeon RX 7900 XTX', vramTotal: 8 * GIB }), 'Radeon RX 7900 XTX · 8 GB');
  assert.equal(gpuLabel({ ...gpu, type: 'cpu' }), 'No GPU (running on CPU)');
  assert.equal(formatVram(7.99 * GIB), '8 GB');
  assert.equal(formatVram(11.6 * GIB), '12 GB');
});
