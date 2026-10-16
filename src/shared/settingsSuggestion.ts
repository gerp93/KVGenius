import { profileFamily } from './modelFamilies';
import { SamplerSettings } from './modelProfiles';

/** File names that say a model is distilled (few steps, little or no guidance). */
const DISTILLED_NAME = /turbo|lightning|lcm|distill|hyper|dmd|schnell/i;

/** Whether a model file's name says it is a distilled model. */
export function isDistilledName(fileName: string): boolean {
  return DISTILLED_NAME.test(fileName);
}

export interface SettingsSuggestion {
  settings: SamplerSettings;
  /** Why these values, in a sentence for the settings step. */
  reason: string;
}

/** What a model that is not distilled starts from: a distilled model's few steps and CFG 1 would leave it unfinished. */
const FULL_MODEL_STEPS = 30;
const FULL_MODEL_CFG = 4;

/**
 * The sampler settings to start a model from, worked out from the family and the files picked - the same files always give the same
 * answer. A family's own defaults are those of its built-in (distilled) model, so they suit a file named like one; any other file
 * keeps the family's sampler, scheduler and shift and gets the steps and CFG a full model needs. Null for a family whose sampler the
 * app does not drive (Wan: Fast / High quality is chosen when generating).
 */
export function suggestSettings(family: string, files: Record<string, string>): SettingsSuggestion | null {
  const def = profileFamily(family);
  if (!def?.sampler) return null;
  const model = files[def.slots[0].key] ?? '';
  if (model === '' || isDistilledName(model)) {
    return {
      settings: { ...def.sampler },
      reason: model === '' ? `The starting values of ${def.builtInName}, which is distilled.` : `The file name says it is distilled, like ${def.builtInName}, so its settings suit it.`,
    };
  }
  return {
    settings: { ...def.sampler, steps: FULL_MODEL_STEPS, cfg: FULL_MODEL_CFG },
    reason: `The file name does not say it is distilled, so it probably needs more steps and real guidance than ${def.builtInName} (${def.sampler.steps} steps, CFG ${def.sampler.cfg}). This is a common starting point; the model's page says what it recommends.`,
  };
}
