/**
 * Every model file the curated templates in src/main/templates/ ask ComfyUI for, by name.
 * This is plain data (a link rots, a file gets renamed upstream - fix it here, not in code).
 * `modelManifest.test.ts` checks it against the template JSON, so the two cannot drift: if a
 * template's loader file changes, that test fails until this list matches.
 *
 * Only files the app itself needs are listed - not a catalog. Other models come in as profiles.
 */

/** ComfyUI's own folder under `models/` for each kind of file. */
export type ModelFolder = 'diffusion_models' | 'text_encoders' | 'vae' | 'loras' | 'upscale_models';

export const MODEL_FOLDERS: readonly ModelFolder[] = ['diffusion_models', 'text_encoders', 'vae', 'loras', 'upscale_models'];

/** The ComfyUI loader node (and its file input) that lists the files in each folder. */
export const FOLDER_LOADERS: Record<ModelFolder, { node: string; input: string }> = {
  diffusion_models: { node: 'UNETLoader', input: 'unet_name' },
  text_encoders: { node: 'CLIPLoader', input: 'clip_name' },
  vae: { node: 'VAELoader', input: 'vae_name' },
  loras: { node: 'LoraLoaderModelOnly', input: 'lora_name' },
  upscale_models: { node: 'UpscaleModelLoader', input: 'model_name' },
};

export interface ManifestFile {
  /** The exact file name the template asks ComfyUI for. */
  file: string;
  folder: ModelFolder;
  /** What the file is, in a few words. */
  role: string;
  /** A direct link to the file (https), for the download helper. Absent: it has to be fetched by hand. */
  url?: string;
}

/**
 * ComfyUI's own repackaged copies on Hugging Face keep every file under `split_files/<folder>/` - public, no
 * account or token needed. NOTE: these links follow that layout but have not been opened from the build environment;
 * the download helper reports a failed file (and carries on), so a moved file shows up as a 404 there.
 */
function hf(repo: string, folder: ModelFolder, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/split_files/${folder}/${file}`;
}

const Z_IMAGE_REPO = 'Comfy-Org/z_image_turbo';
const WAN_REPO = 'Comfy-Org/Wan_2.2_ComfyUI_Repackaged';
const QWEN_IMAGE_REPO = 'Comfy-Org/Qwen-Image_ComfyUI';
const HIDREAM_REPO = 'Comfy-Org/HiDream-I1_ComfyUI';
const OVIS_REPO = 'Comfy-Org/Ovis-Image';

export interface ManifestFeature {
  id: string;
  title: string;
  /** What the user can make with it. */
  summary: string;
  /** The template family that uses these files (null: no fixed files, the user chooses). */
  family: string | null;
  files: ManifestFile[];
  /** Where to read about the model and get its files. */
  source: { label: string; url: string };
  /** Anything worth saying that the table cannot. */
  note?: string;
}

export const MODEL_MANIFEST: ManifestFeature[] = [
  {
    id: 'z-image',
    title: 'Images (Z Image Turbo)',
    summary: 'Text to image, and image to image (starting from an image of your own).',
    family: 'z-image',
    files: [
      { file: 'z_image_turbo_bf16.safetensors', folder: 'diffusion_models', role: 'Image model', url: hf(Z_IMAGE_REPO, 'diffusion_models', 'z_image_turbo_bf16.safetensors') },
      { file: 'qwen_3_4b.safetensors', folder: 'text_encoders', role: 'Text encoder', url: hf(Z_IMAGE_REPO, 'text_encoders', 'qwen_3_4b.safetensors') },
      { file: 'ae.safetensors', folder: 'vae', role: 'VAE', url: hf(Z_IMAGE_REPO, 'vae', 'ae.safetensors') },
    ],
    source: { label: 'Z Image Turbo model page', url: 'https://docs.comfy.org/tutorials/image/z-image/z-image-turbo' },
  },
  {
    id: 'qwen-image',
    title: 'Images (Qwen-Image)',
    summary: 'Text to image. Strong at text inside the picture and at long, detailed prompts; slower than Z Image.',
    family: 'qwen-image',
    files: [
      { file: 'qwen_image_2512_fp8_e4m3fn.safetensors', folder: 'diffusion_models', role: 'Image model', url: hf(QWEN_IMAGE_REPO, 'diffusion_models', 'qwen_image_2512_fp8_e4m3fn.safetensors') },
      { file: 'qwen_2.5_vl_7b_fp8_scaled.safetensors', folder: 'text_encoders', role: 'Text encoder', url: hf(QWEN_IMAGE_REPO, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors') },
      { file: 'qwen_image_vae.safetensors', folder: 'vae', role: 'VAE', url: hf(QWEN_IMAGE_REPO, 'vae', 'qwen_image_vae.safetensors') },
    ],
    source: { label: 'Qwen-Image files (Hugging Face)', url: 'https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI' },
    note: 'The original Qwen-Image model (qwen_image_fp8_e4m3fn.safetensors) works in the same slots: add it as a model on the Models page.',
  },
  {
    id: 'flux-dev',
    title: 'Images (Flux.1 dev)',
    summary: 'Text to image. High detail and good prompt following; slower than Z Image.',
    family: 'flux-dev',
    files: [
      { file: 'flux1-dev.safetensors', folder: 'diffusion_models', role: 'Image model', url: 'https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev.safetensors' },
      { file: 'clip_l.safetensors', folder: 'text_encoders', role: 'CLIP-L text encoder', url: 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors' },
      { file: 't5xxl_fp16.safetensors', folder: 'text_encoders', role: 'T5 text encoder', url: 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors' },
      { file: 'ae.safetensors', folder: 'vae', role: 'VAE', url: hf(Z_IMAGE_REPO, 'vae', 'ae.safetensors') },
    ],
    source: { label: 'Flux.1 dev files (Hugging Face)', url: 'https://huggingface.co/Comfy-Org/flux1-dev' },
    note: 'The VAE is the same file Z Image uses. The model is released under a non-commercial licence.',
  },
  {
    id: 'hidream',
    title: 'Images (HiDream I1 Full)',
    summary: 'Text to image with four text encoders. A large model: the slowest picture family here.',
    family: 'hidream',
    files: [
      { file: 'hidream_i1_full_fp8.safetensors', folder: 'diffusion_models', role: 'Image model', url: hf(HIDREAM_REPO, 'diffusion_models', 'hidream_i1_full_fp8.safetensors') },
      { file: 'clip_l_hidream.safetensors', folder: 'text_encoders', role: 'CLIP-L text encoder', url: hf(HIDREAM_REPO, 'text_encoders', 'clip_l_hidream.safetensors') },
      { file: 'clip_g_hidream.safetensors', folder: 'text_encoders', role: 'CLIP-G text encoder', url: hf(HIDREAM_REPO, 'text_encoders', 'clip_g_hidream.safetensors') },
      { file: 't5xxl_fp8_e4m3fn_scaled.safetensors', folder: 'text_encoders', role: 'T5 text encoder', url: hf(HIDREAM_REPO, 'text_encoders', 't5xxl_fp8_e4m3fn_scaled.safetensors') },
      { file: 'llama_3.1_8b_instruct_fp8_scaled.safetensors', folder: 'text_encoders', role: 'Llama text encoder', url: hf(HIDREAM_REPO, 'text_encoders', 'llama_3.1_8b_instruct_fp8_scaled.safetensors') },
      { file: 'ae.safetensors', folder: 'vae', role: 'VAE', url: hf(Z_IMAGE_REPO, 'vae', 'ae.safetensors') },
    ],
    source: { label: 'HiDream I1 files (Hugging Face)', url: 'https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI' },
    note: 'Needs all four text encoders. The VAE is the same file Z Image uses.',
  },
  {
    id: 'ovis-image',
    title: 'Images (Ovis-Image)',
    summary: 'Text to image. Good at rendering text inside the picture.',
    family: 'ovis-image',
    files: [
      { file: 'ovis_image_bf16.safetensors', folder: 'diffusion_models', role: 'Image model', url: hf(OVIS_REPO, 'diffusion_models', 'ovis_image_bf16.safetensors') },
      { file: 'ovis_2.5.safetensors', folder: 'text_encoders', role: 'Text encoder', url: hf(OVIS_REPO, 'text_encoders', 'ovis_2.5.safetensors') },
      { file: 'ae.safetensors', folder: 'vae', role: 'VAE', url: hf(Z_IMAGE_REPO, 'vae', 'ae.safetensors') },
    ],
    source: { label: 'Ovis-Image files (Hugging Face)', url: 'https://huggingface.co/Comfy-Org/Ovis-Image' },
    note: 'The VAE is the same file Z Image uses.',
  },
  {
    id: 'wan22-i2v',
    title: 'Video (Wan 2.2 image to video)',
    summary: 'Makes a short video from an image.',
    family: 'wan22-i2v',
    files: [
      { file: 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors', folder: 'diffusion_models', role: 'Video model, high noise', url: hf(WAN_REPO, 'diffusion_models', 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors') },
      { file: 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors', folder: 'diffusion_models', role: 'Video model, low noise', url: hf(WAN_REPO, 'diffusion_models', 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors') },
      { file: 'umt5_xxl_fp8_e4m3fn_scaled.safetensors', folder: 'text_encoders', role: 'Text encoder', url: hf(WAN_REPO, 'text_encoders', 'umt5_xxl_fp8_e4m3fn_scaled.safetensors') },
      { file: 'wan_2.1_vae.safetensors', folder: 'vae', role: 'VAE', url: hf(WAN_REPO, 'vae', 'wan_2.1_vae.safetensors') },
      { file: 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors', folder: 'loras', role: '4-step LoRA, high noise (Fast quality)', url: hf(WAN_REPO, 'loras', 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors') },
      { file: 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors', folder: 'loras', role: '4-step LoRA, low noise (Fast quality)', url: hf(WAN_REPO, 'loras', 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors') },
    ],
    source: { label: 'Wan 2.2 files (Hugging Face)', url: 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged' },
    note: 'The workflow contains both LoRAs, so install all six files even if you only use High quality.',
  },
  {
    id: 'wan22-t2v',
    title: 'Video (Wan 2.2 text to video)',
    summary: 'Makes a short video from a prompt alone. Shares the text encoder and VAE with image to video.',
    family: 'wan22-t2v',
    files: [
      { file: 'wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors', folder: 'diffusion_models', role: 'Video model, high noise', url: hf(WAN_REPO, 'diffusion_models', 'wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors') },
      { file: 'wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors', folder: 'diffusion_models', role: 'Video model, low noise', url: hf(WAN_REPO, 'diffusion_models', 'wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors') },
      { file: 'umt5_xxl_fp8_e4m3fn_scaled.safetensors', folder: 'text_encoders', role: 'Text encoder', url: hf(WAN_REPO, 'text_encoders', 'umt5_xxl_fp8_e4m3fn_scaled.safetensors') },
      { file: 'wan_2.1_vae.safetensors', folder: 'vae', role: 'VAE', url: hf(WAN_REPO, 'vae', 'wan_2.1_vae.safetensors') },
      { file: 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors', folder: 'loras', role: '4-step LoRA, high noise (Fast quality)', url: hf(WAN_REPO, 'loras', 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors') },
      { file: 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors', folder: 'loras', role: '4-step LoRA, low noise (Fast quality)', url: hf(WAN_REPO, 'loras', 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors') },
    ],
    source: { label: 'Wan 2.2 files (Hugging Face)', url: 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged' },
    note: 'Text to video needs its own pair of video models and LoRAs; the text encoder and VAE are the same files image to video uses.',
  },
  {
    id: 'upscale',
    title: 'Upscaling',
    summary: 'Enlarges images and videos. You choose the model - any ESRGAN-style file works (.pth or .safetensors).',
    family: null,
    files: [],
    source: { label: 'Browse upscale models', url: 'https://openmodeldb.info' },
  },
];

export function manifestFeature(id: string): ManifestFeature | undefined {
  return MODEL_MANIFEST.find((feature) => feature.id === id);
}
