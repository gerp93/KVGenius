import { GenerationParams } from '../shared/types';
import { DEFAULT_DENOISE, I2I_FAMILY } from '../shared/imageToImage';
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
export function fillImageToImage(workflow: Workflow, params: GenerationParams, uploadedName: string): void {
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
  if (params.modelSettings) applyModelSettings(workflow, I2I_FAMILY, params.modelSettings);
}
