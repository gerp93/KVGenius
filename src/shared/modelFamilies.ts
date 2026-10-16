import { I2I_FAMILY, INPAINT_FAMILY, OUTPAINT_FAMILY } from './imageToImage';
import { V2V_FAMILY } from './videoToVideo';
import type { ModelFolder } from './modelManifest';

/**
 * What a model profile can change within a family: which file goes in each loader slot, and (for
 * families whose sampler the app drives) the sampler settings. The family still decides the graph -
 * these are only values inside it. The node ids behind each slot live with the template patching
 * (main/modelPatch.ts); `modelProfiles.test.ts` checks the defaults here against the template files.
 */
export interface ModelSlot {
  /** Stable key stored in profiles and on generation records. */
  key: string;
  label: string;
  /** Where the file lives under ComfyUI's models folder - decides which installed files are offered. */
  folder: ModelFolder;
  /** The file the shipped template loads (the built-in profile). */
  defaultFile: string;
}

export interface SamplerDefaults {
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  /** ModelSamplingAuraFlow's shift. */
  shift: number;
}

export interface ProfileFamily {
  family: string;
  label: string;
  /** Name of the built-in profile - the shipped template as it is. */
  builtInName: string;
  slots: ModelSlot[];
  /** null: the app does not drive this family's sampler (a profile carries files only). */
  sampler: SamplerDefaults | null;
  /** What the CFG field means here, when it is not classifier-free guidance (Flux: its distilled guidance value). */
  cfgLabel?: string;
  /** True when the workflow has no shift to set (Flux), so the field is not offered. */
  noShift?: boolean;
}

export const WAN_FAMILY = {
  family: 'wan22-i2v',
  label: 'Wan 2.2 (image to video)',
  builtInName: 'Wan 2.2 image to video',
  slots: [
    { key: 'highNoiseModel', label: 'Video model, high noise', folder: 'diffusion_models', defaultFile: 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors' },
    { key: 'lowNoiseModel', label: 'Video model, low noise', folder: 'diffusion_models', defaultFile: 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors' },
    { key: 'textEncoder', label: 'Text encoder', folder: 'text_encoders', defaultFile: 'umt5_xxl_fp8_e4m3fn_scaled.safetensors' },
    { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'wan_2.1_vae.safetensors' },
    { key: 'highNoiseLora', label: '4-step LoRA, high noise (Fast quality)', folder: 'loras', defaultFile: 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors' },
    { key: 'lowNoiseLora', label: '4-step LoRA, low noise (Fast quality)', folder: 'loras', defaultFile: 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors' },
  ],
  // The app does not drive Wan's sampler: the Fast / High quality choice on Generate decides steps and CFG.
  sampler: null,
} satisfies ProfileFamily;

export const WAN_T2V_FAMILY = {
  family: 'wan22-t2v',
  label: 'Wan 2.2 (text to video)',
  builtInName: 'Wan 2.2 text to video',
  slots: [
    { key: 'highNoiseModel', label: 'Video model, high noise', folder: 'diffusion_models', defaultFile: 'wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors' },
    { key: 'lowNoiseModel', label: 'Video model, low noise', folder: 'diffusion_models', defaultFile: 'wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors' },
    { key: 'textEncoder', label: 'Text encoder', folder: 'text_encoders', defaultFile: 'umt5_xxl_fp8_e4m3fn_scaled.safetensors' },
    { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'wan_2.1_vae.safetensors' },
    { key: 'highNoiseLora', label: '4-step LoRA, high noise (Fast quality)', folder: 'loras', defaultFile: 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors' },
    { key: 'lowNoiseLora', label: '4-step LoRA, low noise (Fast quality)', folder: 'loras', defaultFile: 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors' },
  ],
  sampler: null,
} satisfies ProfileFamily;

export const QWEN_IMAGE_FAMILY = {
  family: 'qwen-image',
  label: 'Qwen-Image (text to image)',
  builtInName: 'Qwen-Image',
  slots: [
    { key: 'diffusionModel', label: 'Image model', folder: 'diffusion_models', defaultFile: 'qwen_image_2512_fp8_e4m3fn.safetensors' },
    { key: 'textEncoder', label: 'Text encoder', folder: 'text_encoders', defaultFile: 'qwen_2.5_vl_7b_fp8_scaled.safetensors' },
    { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'qwen_image_vae.safetensors' },
  ],
  // ComfyUI's own Qwen-Image workflow (not its Lightning one): 50 steps at CFG 4, shift 3.1.
  sampler: { steps: 50, cfg: 4, sampler: 'euler', scheduler: 'simple', shift: 3.1 },
} satisfies ProfileFamily;

export const FLUX_DEV_FAMILY = {
  family: 'flux-dev',
  label: 'Flux.1 dev (text to image)',
  builtInName: 'Flux.1 dev',
  slots: [
    { key: 'diffusionModel', label: 'Image model', folder: 'diffusion_models', defaultFile: 'flux1-dev.safetensors' },
    { key: 'clipL', label: 'CLIP-L text encoder', folder: 'text_encoders', defaultFile: 'clip_l.safetensors' },
    { key: 'textEncoder', label: 'T5 text encoder', folder: 'text_encoders', defaultFile: 't5xxl_fp16.safetensors' },
    { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'ae.safetensors' },
  ],
  // Flux dev is guidance-distilled: the sampler's CFG stays 1 and the "CFG" here is the FluxGuidance value (3.5 is its default).
  sampler: { steps: 20, cfg: 3.5, sampler: 'euler', scheduler: 'simple', shift: 0 },
  cfgLabel: 'Guidance',
  noShift: true,
} satisfies ProfileFamily;

export const HIDREAM_FAMILY = {
  family: 'hidream',
  label: 'HiDream I1 (text to image)',
  builtInName: 'HiDream I1 Full',
  slots: [
    { key: 'diffusionModel', label: 'Image model', folder: 'diffusion_models', defaultFile: 'hidream_i1_full_fp8.safetensors' },
    { key: 'clipL', label: 'CLIP-L text encoder', folder: 'text_encoders', defaultFile: 'clip_l_hidream.safetensors' },
    { key: 'clipG', label: 'CLIP-G text encoder', folder: 'text_encoders', defaultFile: 'clip_g_hidream.safetensors' },
    { key: 'textEncoder', label: 'T5 text encoder', folder: 'text_encoders', defaultFile: 't5xxl_fp8_e4m3fn_scaled.safetensors' },
    { key: 'llama', label: 'Llama text encoder', folder: 'text_encoders', defaultFile: 'llama_3.1_8b_instruct_fp8_scaled.safetensors' },
    { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'ae.safetensors' },
  ],
  sampler: { steps: 50, cfg: 5, sampler: 'uni_pc', scheduler: 'simple', shift: 3 },
} satisfies ProfileFamily;

export const OVIS_IMAGE_FAMILY = {
  family: 'ovis-image',
  label: 'Ovis-Image (text to image)',
  builtInName: 'Ovis-Image',
  slots: [
    { key: 'diffusionModel', label: 'Image model', folder: 'diffusion_models', defaultFile: 'ovis_image_bf16.safetensors' },
    { key: 'textEncoder', label: 'Text encoder', folder: 'text_encoders', defaultFile: 'ovis_2.5.safetensors' },
    { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'ae.safetensors' },
  ],
  sampler: { steps: 20, cfg: 5, sampler: 'euler', scheduler: 'simple', shift: 3 },
} satisfies ProfileFamily;

/** What a profile of a family with no app-driven sampler stores in its (never used) sampler columns. */
export const NO_SAMPLER: SamplerDefaults = { steps: 1, cfg: 0, sampler: '', scheduler: '', shift: 0 };

export const PROFILE_FAMILIES: ProfileFamily[] = [
  {
    family: 'z-image',
    label: 'Z-Image (text to image)',
    builtInName: 'Z Image Turbo',
    slots: [
      { key: 'diffusionModel', label: 'Image model', folder: 'diffusion_models', defaultFile: 'z_image_turbo_bf16.safetensors' },
      { key: 'textEncoder', label: 'Text encoder', folder: 'text_encoders', defaultFile: 'qwen_3_4b.safetensors' },
      { key: 'vae', label: 'VAE', folder: 'vae', defaultFile: 'ae.safetensors' },
    ],
    sampler: { steps: 8, cfg: 1, sampler: 'res_multistep', scheduler: 'simple', shift: 3 },
  },
  QWEN_IMAGE_FAMILY,
  FLUX_DEV_FAMILY,
  HIDREAM_FAMILY,
  OVIS_IMAGE_FAMILY,
  WAN_FAMILY,
  WAN_T2V_FAMILY,
];

/** Families that run another family's model files, and so use that family's model profiles (image to image
 * loads the same Z-Image files as text to image). Their templates must keep the same loader node ids. */
const PROFILE_BASE: Record<string, string> = { [I2I_FAMILY]: 'z-image', [INPAINT_FAMILY]: 'z-image', [OUTPAINT_FAMILY]: 'z-image', [V2V_FAMILY]: 'wan22-t2v' };

/** The family whose profiles apply to a job of `family`. */
export function profileFamilyKey(family: string): string {
  return PROFILE_BASE[family] ?? family;
}

export function profileFamily(family: string): ProfileFamily | undefined {
  return PROFILE_FAMILIES.find((f) => f.family === family);
}

/** Limits a profile's sampler values must stay within (a guard against typos, not a recommendation). */
export const SAMPLER_LIMITS = {
  steps: { min: 1, max: 100 },
  cfg: { min: 0, max: 30 },
  shift: { min: 0, max: 20 },
} as const;

/** Upscale models are not a profile kind (one file, chosen per run on the Upscale page), but a file can still be imported
 * into `upscale_models` the same way as any model file: this is the pseudo family and its one slot. */
export const UPSCALE_IMPORT_FAMILY = 'upscale';
export const UPSCALE_IMPORT_SLOT: ModelSlot = { key: 'model', label: 'Upscale model', folder: 'upscale_models', defaultFile: '' };

/** The loader slot a model file is being imported for: one of a profile family's, or the upscale model's. */
export function importSlot(family: unknown, slotKey: unknown): ModelSlot | undefined {
  if (family === UPSCALE_IMPORT_FAMILY) return slotKey === UPSCALE_IMPORT_SLOT.key ? UPSCALE_IMPORT_SLOT : undefined;
  return typeof family === 'string' ? profileFamily(family)?.slots.find((s) => s.key === slotKey) : undefined;
}
