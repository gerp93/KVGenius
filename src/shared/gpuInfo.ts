/** One compute device as ComfyUI reports it (`/system_stats` -> `devices`). */
export interface GpuInfo {
  /** Cleaned up for display, e.g. "NVIDIA GeForce RTX 4090". */
  name: string;
  /** 'cuda', 'mps', 'cpu', ... as ComfyUI names it. */
  type: string;
  vramTotal: number;
  vramFree: number;
}

type RawDevice = { name?: unknown; type?: unknown; vram_total?: unknown; vram_free?: unknown };

/** ComfyUI names a device like "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync"; keep just the card's name. */
export function cleanDeviceName(raw: string): string {
  return raw
    .replace(/^[a-z]+:\d+\s+/i, '')
    .replace(/\s+:\s+.*$/, '')
    .trim();
}

/** The devices in a `/system_stats` response, tolerating anything missing or the wrong shape. */
export function parseSystemStats(data: unknown): GpuInfo[] {
  const devices = (data as { devices?: unknown } | null)?.devices;
  if (!Array.isArray(devices)) return [];
  const found: GpuInfo[] = [];
  for (const raw of devices as RawDevice[]) {
    if (!raw || typeof raw.name !== 'string') continue;
    found.push({
      name: cleanDeviceName(raw.name) || raw.name,
      type: typeof raw.type === 'string' ? raw.type : '',
      vramTotal: typeof raw.vram_total === 'number' ? raw.vram_total : 0,
      vramFree: typeof raw.vram_free === 'number' ? raw.vram_free : 0,
    });
  }
  return found;
}

const GIB = 1024 ** 3;

/** "24 GB" - whole gigabytes, one decimal below 10 (so an 8 GB card does not read 7.99). */
export function formatVram(bytes: number): string {
  const gb = bytes / GIB;
  return `${gb >= 10 ? Math.round(gb) : Math.round(gb * 10) / 10} GB`;
}

/** The one line shown in the sidebar: "RTX 4090 - 24 GB" (the vendor prefix is dropped to save room). */
export function gpuLabel(gpu: GpuInfo): string {
  if (gpu.type === 'cpu') return 'No GPU (running on CPU)';
  const short = gpu.name.replace(/^(NVIDIA|AMD)\s+/i, '').replace(/^GeForce\s+/i, '');
  return gpu.vramTotal > 0 ? `${short} · ${formatVram(gpu.vramTotal)}` : short;
}
