import { ModelSettings } from '../shared/modelProfiles';
import { profileFamily } from '../shared/modelFamilies';

type Workflow = Record<string, { inputs: Record<string, unknown> } | undefined>;

/** Which template node and input each profile slot sets (node ids are those of src/main/templates/<family>.json). */
export const SLOT_NODES: Record<string, Record<string, { node: string; input: string }>> = {
  'z-image': {
    diffusionModel: { node: '57:28', input: 'unet_name' },
    textEncoder: { node: '57:30', input: 'clip_name' },
    vae: { node: '57:29', input: 'vae_name' },
  },
};

/** The sampler nodes a profile sets (KSampler's sampler_name / scheduler, ModelSamplingAuraFlow's shift). */
export const SAMPLER_NODES: Record<string, { sampler: string; shift: string }> = {
  'z-image': { sampler: '57:3', shift: '57:11' },
};

function nodeInputs(workflow: Workflow, node: string): Record<string, unknown> {
  const found = workflow[node];
  if (!found) throw new Error(`The workflow template has no node ${node} - the model patch is out of date with it.`);
  return found.inputs;
}

/**
 * Writes a profile's files and sampler values into a copy of the template. Steps and CFG are set by the caller
 * (they travel in the job params). Throws when a slot has no node, so a template that drifts from this map
 * fails loudly instead of silently running the wrong model.
 */
export function applyModelSettings(workflow: Workflow, family: string, settings: ModelSettings): void {
  const def = profileFamily(family);
  const slots = SLOT_NODES[family];
  const sampler = SAMPLER_NODES[family];
  if (!def || !slots || !sampler) throw new Error(`Model profiles are not supported for '${family}'.`);
  for (const slot of def.slots) {
    const file = settings.files[slot.key];
    if (!file) throw new Error(`The model settings have no ${slot.label.toLowerCase()} file.`);
    const target = slots[slot.key];
    nodeInputs(workflow, target.node)[target.input] = file;
  }
  const samplerInputs = nodeInputs(workflow, sampler.sampler);
  samplerInputs.sampler_name = settings.sampler;
  samplerInputs.scheduler = settings.scheduler;
  nodeInputs(workflow, sampler.shift).shift = settings.shift;
}
