import type { GenerationParams } from '../shared/types';

type Workflow = Record<string, { inputs: Record<string, unknown> } | undefined>;

/**
 * Node IDs in src/main/templates/z-image.json that fillTextToImage() fills in.
 * If that file changes, keep these in sync.
 */
const Z_IMAGE_TURBO_NODE_MAP = {
  prompt: '57:27',
  sampler: '57:3',
  latent: '57:13',
};

/**
 * The same fields for the newer text-to-image families (qwen-image, flux-dev, hidream, ovis-image): all four templates use the same
 * node ids. `GUIDANCE_NODE` is set only where the "CFG" is a FluxGuidance value rather than the sampler's CFG.
 */
const TEXT_TO_IMAGE_NODE_MAP = { prompt: '6', sampler: '10', latent: '9' };
const GUIDANCE_NODE: Record<string, string> = { 'flux-dev': '7' };
const NODE_MAP_FOR: Record<string, { prompt: string; sampler: string; latent: string }> = {
  'qwen-image': TEXT_TO_IMAGE_NODE_MAP,
  'flux-dev': TEXT_TO_IMAGE_NODE_MAP,
  hidream: TEXT_TO_IMAGE_NODE_MAP,
  'ovis-image': TEXT_TO_IMAGE_NODE_MAP,
};

function inputsOf(workflow: Workflow, node: string): Record<string, unknown> {
  const found = workflow[node];
  if (!found) throw new Error(`The workflow template has no node ${node} - the patch is out of date with it.`);
  return found.inputs;
}

/** Puts the prompt, seed, steps, CFG and size of a text-to-image job into a copy of its family's template (Z-Image's unless the family has its own). */
export function fillTextToImage(workflow: Workflow, family: string, params: GenerationParams): void {
  const nodes = NODE_MAP_FOR[family] ?? Z_IMAGE_TURBO_NODE_MAP;
  inputsOf(workflow, nodes.prompt).text = params.prompt;

  const sampler = inputsOf(workflow, nodes.sampler);
  sampler.seed = params.seed;
  sampler.steps = params.steps;
  const guidanceNode = GUIDANCE_NODE[family];
  if (guidanceNode) {
    // The sampler's own CFG stays as the template has it (1); the user's CFG value is the model's guidance.
    inputsOf(workflow, guidanceNode).guidance = params.cfg;
  } else {
    sampler.cfg = params.cfg;
  }

  const latent = inputsOf(workflow, nodes.latent);
  latent.width = params.width;
  latent.height = params.height;
}
