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
 * Families whose own defaults are those of the full model (the settings of ComfyUI's workflow for it), not of a distilled one - the
 * opposite of Z-Image, whose built-in is Turbo. A file named like a distilled model gets a few steps instead.
 */
const FULL_DEFAULT_FAMILIES = new Set(['qwen-image', 'flux-dev', 'hidream', 'ovis-image']);

/** A few steps and little guidance: where a distilled model of a full-default family starts (Lightning LoRAs for Qwen-Image are 4 or 8 steps). */
const DISTILLED_STEPS = 8;
const DISTILLED_CFG = 1;

/** Releases of one family that have their own settings and are told apart by file name (HiDream's Dev and Fast are distilled from Full; Flux schnell takes 4 steps). */
const FAMILY_VARIANTS: Record<string, { match: RegExp; label: string; settings: Partial<SamplerSettings> }[]> = {
  'flux-dev': [{ match: /schnell/i, label: 'Flux schnell', settings: { steps: 4 } }],
  hidream: [
    { match: /fast/i, label: 'HiDream Fast', settings: { steps: 16, cfg: 1, sampler: 'lcm', scheduler: 'normal', shift: 3 } },
    { match: /dev/i, label: 'HiDream Dev', settings: { steps: 28, cfg: 1, sampler: 'lcm', scheduler: 'normal', shift: 6 } },
  ],
};

/** "4steps" / "8-steps" / "2 step" in a file name, when it says how many steps it was made for. */
function stepsInName(fileName: string): number | null {
  const found = /(\d{1,2})[\s_-]*steps?/i.exec(fileName);
  const steps = found ? Number(found[1]) : NaN;
  return Number.isInteger(steps) && steps >= 1 ? steps : null;
}

/**
 * The sampler settings to start a model from, worked out from the family and the files picked - the same files always give the same
 * answer. For Z-Image the family's own defaults are those of its built-in (distilled) model, so they suit a file named like one; any
 * other file keeps the family's sampler, scheduler and shift and gets the steps and CFG a full model needs. The newer families'
 * defaults are the full model's, so only a file named like a distilled model changes them. Null for a family whose sampler the
 * app does not drive (Wan: Fast / High quality is chosen when generating).
 */
export function suggestSettings(family: string, files: Record<string, string>): SettingsSuggestion | null {
  const def = profileFamily(family);
  if (!def?.sampler) return null;
  const model = files[def.slots[0].key] ?? '';

  if (FULL_DEFAULT_FAMILIES.has(family)) {
    const variant = FAMILY_VARIANTS[family]?.find((v) => v.match.test(model));
    if (variant) {
      return { settings: { ...def.sampler, ...variant.settings }, reason: `The file name says it is ${variant.label}, which ComfyUI's workflow runs with these values.` };
    }
    if (model !== '' && isDistilledName(model)) {
      const steps = stepsInName(model) ?? DISTILLED_STEPS;
      return {
        settings: { ...def.sampler, steps, cfg: family === 'flux-dev' ? def.sampler.cfg : DISTILLED_CFG },
        reason: `The file name says it is distilled, so it needs only a few steps${stepsInName(model) ? ` (${steps}, as its name says)` : ''} instead of ${def.builtInName}'s ${def.sampler.steps}.`,
      };
    }
    return {
      settings: { ...def.sampler },
      reason: model === '' ? `The starting values of ${def.builtInName}.` : `The values ComfyUI's own workflow uses for ${def.builtInName}. A fine-tune may recommend others; its model page says.`,
    };
  }

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
