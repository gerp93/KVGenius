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
  WAN_FAMILY,
];

export function profileFamily(family: string): ProfileFamily | undefined {
  return PROFILE_FAMILIES.find((f) => f.family === family);
}

/** Limits a profile's sampler values must stay within (a guard against typos, not a recommendation). */
export const SAMPLER_LIMITS = {
  steps: { min: 1, max: 100 },
  cfg: { min: 0, max: 30 },
  shift: { min: 0, max: 20 },
} as const;
