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
