import { GenerationParams } from '../shared/types';
import { v2vSchedule, V2V_DEFAULT_STRENGTH } from '../shared/videoToVideo';
import { videoQualityFromCfg } from '../shared/videoQuality';
import { applyModelSettings } from './modelPatch';

type Workflow = Record<string, { inputs: Record<string, unknown> } | undefined>;

/**
 * Node IDs in src/main/templates/wan22-v2v.json: the text-to-video graph (same ids for the loaders, prompt, samplers and the Fast /
 * High switch, so model profiles apply unchanged) with the empty latent replaced by
 * load video -> split into frames -> scale to the output size -> first N frames -> VAE encode. The saved video takes its
 * frame rate from the source.
 */
export const WAN22_V2V_NODE_MAP = {
  loadVideo: 'v2v-load',
  scale: 'v2v-scale',
  frames: 'v2v-frames',
  prompt: '129:93',
  samplerHighNoise: '129:86',
  samplerLowNoise: '129:85',
  fastLoraSwitch: '129:131',
};

function inputs(workflow: Workflow, node: string): Record<string, unknown> {
  const found = workflow[node];
  if (!found) throw new Error(`The video-to-video template has no node ${node} - the patch is out of date with it.`);
  return found.inputs;
}

/**
 * Fills the video-to-video template in place: the source video (already uploaded to ComfyUI as `uploadedName`), the output size
 * and how many of its first frames to use, the prompt, the seed, the Fast / High switch and the strength - and, if a model
 * profile is in use, its files.
 */
export function fillVideoToVideo(workflow: Workflow, params: GenerationParams, uploadedName: string, family: string): void {
  const nodes = WAN22_V2V_NODE_MAP;
  inputs(workflow, nodes.loadVideo).file = uploadedName;
  const scale = inputs(workflow, nodes.scale);
  scale.width = params.width;
  scale.height = params.height;
  inputs(workflow, nodes.frames).length = params.length ?? 81;
  inputs(workflow, nodes.prompt).text = params.prompt;
  inputs(workflow, nodes.samplerHighNoise).noise_seed = params.seed;
  inputs(workflow, nodes.fastLoraSwitch).value = videoQualityFromCfg(params.cfg) === 'fast';

  // Start part of the way through the schedule instead of at the top, as the strength says.
  const schedule = v2vSchedule(params.denoise ?? V2V_DEFAULT_STRENGTH, params.cfg);
  inputs(workflow, nodes.samplerHighNoise).start_at_step = schedule.firstStart;
  if (schedule.secondAddsNoise && schedule.secondStart !== null) {
    const low = inputs(workflow, nodes.samplerLowNoise);
    low.add_noise = 'enable';
    low.noise_seed = params.seed;
    low.start_at_step = schedule.secondStart;
  }

  if (params.modelSettings) applyModelSettings(workflow, family, params.modelSettings);
}
