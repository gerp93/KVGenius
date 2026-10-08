import { GenerationParams } from '../shared/types';
import { DEFAULT_DENOISE, I2I_FAMILY, INPAINT_FAMILY } from '../shared/imageToImage';
import { applyModelSettings } from './modelPatch';

type Workflow = Record<string, { inputs: Record<string, unknown> } | undefined>;

/**
 * Node IDs in src/main/templates/z-image-i2i.json: the text-to-image graph (same ids for the loaders, prompt,
 * sampler and shift, so model profiles apply unchanged) with the empty latent replaced by
 * load picture -> scale to the output size -> VAE encode.
 */
export const Z_IMAGE_I2I_NODE_MAP = {
  prompt: '57:27',
  sampler: '57:3',
  loadImage: 'i2i-load',
  scale: 'i2i-scale',
  encode: 'i2i-encode',
};

function inputs(workflow: Workflow, node: string): Record<string, unknown> {
  const found = workflow[node];
  if (!found) throw new Error(`The image-to-image template has no node ${node} - the patch is out of date with it.`);
  return found.inputs;
}

/**
 * Fills the image-to-image template in place: the start picture (already uploaded to ComfyUI as `uploadedName`),
 * the output size it is fitted to (cropped from the centre if the shapes differ), the prompt, the sampler values and
 * the strength - and, if a model profile is in use, its files and sampler.
 */
export function fillImageToImage(workflow: Workflow, params: GenerationParams, uploadedName: string, family: string = I2I_FAMILY): void {
  const nodes = Z_IMAGE_I2I_NODE_MAP;
  inputs(workflow, nodes.loadImage).image = uploadedName;
  const scale = inputs(workflow, nodes.scale);
  scale.width = params.width;
  scale.height = params.height;
  inputs(workflow, nodes.prompt).text = params.prompt;
  const sampler = inputs(workflow, nodes.sampler);
  sampler.seed = params.seed;
  sampler.steps = params.steps;
  sampler.cfg = params.cfg;
  sampler.denoise = params.denoise ?? DEFAULT_DENOISE;
  if (params.modelSettings) applyModelSettings(workflow, family, params.modelSettings);
}

/**
 * Node IDs in src/main/templates/z-image-inpaint.json: the image-to-image graph (same ids) plus the mask steps - load the
 * mask, fit it to the output size the same way the picture is fitted, soften its edge, and use it twice: to limit the
 * sampler to the painted spots, and to paste the result back over the original so everything else stays untouched.
 */
export const Z_IMAGE_INPAINT_NODE_MAP = {
  ...Z_IMAGE_I2I_NODE_MAP,
  loadMask: 'ip-maskload',
  maskScale: 'ip-maskscale',
};

/** Fills the inpainting template in place: everything image to image sets, plus the mask (already uploaded as `maskName`). */
export function fillInpaint(workflow: Workflow, params: GenerationParams, imageName: string, maskName: string): void {
  fillImageToImage(workflow, params, imageName, INPAINT_FAMILY);
  const nodes = Z_IMAGE_INPAINT_NODE_MAP;
  inputs(workflow, nodes.loadMask).image = maskName;
  const scale = inputs(workflow, nodes.maskScale);
  scale.width = params.width;
  scale.height = params.height;
}
