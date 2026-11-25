import { LoraSpec, MAX_LORAS, LORA_STRENGTH } from '../shared/loras';

type Workflow = Record<string, { inputs: Record<string, unknown>; class_type?: string; _meta?: { title: string } } | undefined>;

/**
 * Where LoRAs go in a family's template. `model` is the node that loads the diffusion model; `consumers` are the inputs that take its
 * output and must take the last LoRA's instead. Node ids are those of src/main/templates/<family>.json (loraPatch.test.ts checks them).
 *
 * The template itself has no LoRA nodes: ComfyUI checks `lora_name` against the files it has, so a placeholder would make every run
 * without a LoRA fail. The nodes are added here, only for the LoRAs a job actually carries. To let another family load LoRAs, give it an entry
 * here and list it in LORA_FAMILIES (shared/loras.ts).
 */
export const LORA_NODE_MAP: Record<string, { model: string; consumers: { node: string; input: string }[] }> = {
  'flux-dev': { model: '2', consumers: [{ node: '10', input: 'model' }] },
};

/** ComfyUI's core model-only LoRA loader (Flux LoRAs are applied to the diffusion model only). The node the Models page lists LoRA files from. */
export const LORA_LOADER_CLASS = 'LoraLoaderModelOnly';

export function loraNodeId(index: number): string {
  return `lora-${index + 1}`;
}

/**
 * Chains the LoRAs onto the model in a copy of the family's template: model -> LoRA 1 -> LoRA 2 -> ... -> whatever used the model. With none (or a
 * family that cannot load one) the graph is left exactly as it is, with no LoRA node in it. Only the first MAX_LORAS are used.
 * Throws when the template no longer matches LORA_NODE_MAP, so a drift fails loudly instead of silently running without the LoRA.
 */
export function applyLoras(workflow: Workflow, family: string, loras: readonly LoraSpec[] | undefined): void {
  const target = LORA_NODE_MAP[family];
  const chosen = (loras ?? []).filter((l) => l && typeof l.file === 'string' && l.file !== '').slice(0, MAX_LORAS);
  if (!target || chosen.length === 0) return;

  if (!workflow[target.model]) throw new Error(`The workflow template has no node ${target.model} - the LoRA patch is out of date with it.`);
  for (const { node, input } of target.consumers) {
    const link = workflow[node]?.inputs[input];
    if (!Array.isArray(link) || link[0] !== target.model) {
      throw new Error(`Node ${node} of the workflow template no longer takes its ${input} from node ${target.model} - the LoRA patch is out of date with it.`);
    }
  }

  let previous = target.model;
  chosen.forEach((lora, index) => {
    const id = loraNodeId(index);
    const strength = Number.isFinite(lora.strength) ? Math.min(Math.max(lora.strength, LORA_STRENGTH.min), LORA_STRENGTH.max) : LORA_STRENGTH.default;
    workflow[id] = {
      inputs: { model: [previous, 0], lora_name: lora.file, strength_model: strength },
      class_type: LORA_LOADER_CLASS,
      _meta: { title: `LoRA ${index + 1}` },
    };
    previous = id;
  });
  for (const { node, input } of target.consumers) {
    (workflow[node] as { inputs: Record<string, unknown> }).inputs[input] = [previous, 0];
  }
}
