import { GenerationParams } from '../shared/types';
import { FACE_KLEIN_FAMILY, FACE_QWEN_FAMILY } from '../shared/faceReference';

type Workflow = Record<string, { inputs: Record<string, unknown> } | undefined>;

/**
 * Node IDs in src/main/templates/face-klein.json: ComfyUI's Flux.2 Klein 4B (distilled) image-edit graph, with the output
 * size taken from the request instead of from the reference picture. The reference is scaled to about a megapixel, encoded,
 * and attached to both the positive and the (zeroed) negative conditioning as a reference latent. The 4 steps and CFG 1
 * belong to the distilled model and stay as the template has them.
 */
export const FACE_KLEIN_NODE_MAP = {
  loadImage: 'fr-load',
  prompt: 'fr-text',
  noise: 'fr-noise',
  latent: 'fr-latent',
  scheduler: 'fr-sigmas',
};

/**
 * Node IDs in src/main/templates/face-qwen.json: ComfyUI's Qwen-Image-Edit 2511 graph on its 4-step Lightning path. The
 * reference reaches the model through the prompt encoder (it looks at the picture and the prompt together); the sampler starts
 * from an empty latent of the requested size. The positive and negative encoders both carry the reference; only the positive
 * has a prompt.
 */
export const FACE_QWEN_NODE_MAP = {
  loadImage: 'fq-load',
  positive: 'fq-pos',
  sampler: 'fq-sample',
  latent: 'fq-latent',
};

function inputs(workflow: Workflow, node: string, family: string): Record<string, unknown> {
  const found = workflow[node];
  if (!found) throw new Error(`The ${family} template has no node ${node} - the patch is out of date with it.`);
  return found.inputs;
}

/**
 * Fills a face-reference template in place: the reference picture (already uploaded to ComfyUI as `uploadedName`), the
 * prompt, the seed and the output size.
 */
export function fillFaceReference(workflow: Workflow, family: string, params: GenerationParams, uploadedName: string): void {
  if (family === FACE_KLEIN_FAMILY) {
    const nodes = FACE_KLEIN_NODE_MAP;
    inputs(workflow, nodes.loadImage, family).image = uploadedName;
    inputs(workflow, nodes.prompt, family).text = params.prompt;
    inputs(workflow, nodes.noise, family).noise_seed = params.seed;
    const latent = inputs(workflow, nodes.latent, family);
    latent.width = params.width;
    latent.height = params.height;
    const scheduler = inputs(workflow, nodes.scheduler, family);
    scheduler.width = params.width;
    scheduler.height = params.height;
    return;
  }
  if (family === FACE_QWEN_FAMILY) {
    const nodes = FACE_QWEN_NODE_MAP;
    inputs(workflow, nodes.loadImage, family).image = uploadedName;
    inputs(workflow, nodes.positive, family).prompt = params.prompt;
    inputs(workflow, nodes.sampler, family).seed = params.seed;
    const latent = inputs(workflow, nodes.latent, family);
    latent.width = params.width;
    latent.height = params.height;
    return;
  }
  throw new Error(`'${family}' is not a face-reference family.`);
}
