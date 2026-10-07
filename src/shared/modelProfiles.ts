import { NO_SAMPLER, profileFamily, ProfileFamily, SAMPLER_LIMITS } from './modelFamilies';

/** The sampler values a profile sets. */
export interface SamplerSettings {
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  shift: number;
}

/**
 * A saved variant of a model family: a name, a file for each loader slot, and sampler defaults. The family
 * decides the graph; the profile only changes values inside it (see shared/modelFamilies.ts).
 */
export interface ModelProfile {
  id: number;
  family: string;
  name: string;
  /** slot key -> file name, exactly as ComfyUI lists it. */
  files: Record<string, string>;
  sampler: SamplerSettings;
  createdAt: string;
  updatedAt: string;
}

export interface ModelProfileInput {
  family: string;
  name: string;
  files: Record<string, string>;
  sampler: SamplerSettings;
}

/**
 * What a job carries so ComfyUI loads those files and samples that way - the profile resolved at submit
 * time, so a Library record, Re-rack and the queue never depend on the profile still existing. A job with
 * no settings runs the template exactly as shipped. (steps and cfg travel in the params as always.)
 */
export interface ModelSettings {
  files: Record<string, string>;
  /** Only for families whose sampler the app drives (images); a video profile carries files only. */
  sampler?: string;
  scheduler?: string;
  shift?: number;
}

export const MAX_PROFILE_NAME_LENGTH = 60;
const MAX_FILE_NAME_LENGTH = 300;

/** A file name that cannot point outside ComfyUI's models folder: relative, no ".." and no drive or root. */
export function isSafeModelFileName(name: string): boolean {
  if (name.length === 0 || name.length > MAX_FILE_NAME_LENGTH) return false;
  if (/^([a-zA-Z]:|[\\/])/.test(name)) return false;
  return !name.split(/[\\/]/).some((part) => part === '..' || part === '');
}

function inRange(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
}

/** A profile as it will be stored (names trimmed), or the reason it cannot be. */
export function validateProfileInput(input: ModelProfileInput): { ok: true; value: ModelProfileInput } | { ok: false; message: string } {
  const family: ProfileFamily | undefined = typeof input?.family === 'string' ? profileFamily(input.family) : undefined;
  if (!family) return { ok: false, message: 'Choose which kind of model this is.' };

  const name = typeof input.name === 'string' ? input.name.trim() : '';
  if (!name) return { ok: false, message: 'Give the model a name.' };
  if (name.length > MAX_PROFILE_NAME_LENGTH) return { ok: false, message: `The name can be at most ${MAX_PROFILE_NAME_LENGTH} characters.` };
  if (name.toLowerCase() === family.builtInName.toLowerCase()) return { ok: false, message: `"${name}" is the built-in model's name - choose another.` };

  const files: Record<string, string> = {};
  for (const slot of family.slots) {
    const file = typeof input.files?.[slot.key] === 'string' ? input.files[slot.key].trim() : '';
    if (!file) return { ok: false, message: `Choose the ${slot.label.toLowerCase()} file.` };
    if (!isSafeModelFileName(file)) return { ok: false, message: `"${file}" is not a valid file name.` };
    files[slot.key] = file;
  }

  // A family whose sampler the app does not drive (video) has nothing to set here: files only.
  if (!family.sampler) return { ok: true, value: { family: family.family, name, files, sampler: { ...NO_SAMPLER } } };

  const s = input.sampler;
  const { steps, cfg, shift } = SAMPLER_LIMITS;
  if (!inRange(s?.steps, steps.min, steps.max) || !Number.isInteger(s.steps)) return { ok: false, message: `Steps must be a whole number from ${steps.min} to ${steps.max}.` };
  if (!inRange(s.cfg, cfg.min, cfg.max)) return { ok: false, message: `CFG must be from ${cfg.min} to ${cfg.max}.` };
  if (!inRange(s.shift, shift.min, shift.max)) return { ok: false, message: `Shift must be from ${shift.min} to ${shift.max}.` };
  const sampler = typeof s.sampler === 'string' ? s.sampler.trim() : '';
  const scheduler = typeof s.scheduler === 'string' ? s.scheduler.trim() : '';
  if (!sampler || sampler.length > 64) return { ok: false, message: 'Choose a sampler.' };
  if (!scheduler || scheduler.length > 64) return { ok: false, message: 'Choose a scheduler.' };

  return { ok: true, value: { family: family.family, name, files, sampler: { steps: s.steps, cfg: s.cfg, sampler, scheduler, shift: s.shift } } };
}

/** The settings a job carries for this profile. */
export function profileSettings(profile: Pick<ModelProfile, 'family' | 'files' | 'sampler'>): ModelSettings {
  const files = { ...profile.files };
  if (!profileFamily(profile.family)?.sampler) return { files };
  return { files, sampler: profile.sampler.sampler, scheduler: profile.sampler.scheduler, shift: profile.sampler.shift };
}

/** A stable text form of resolved settings (keys sorted), so two runs with the same settings compare equal.
 * null when there are none - the shipped template. */
export function serializeModelSettings(settings: ModelSettings | null | undefined): string | null {
  if (!settings) return null;
  const files = Object.fromEntries(Object.keys(settings.files).sort().map((key) => [key, settings.files[key]]));
  return JSON.stringify({ files, sampler: settings.sampler ?? null, scheduler: settings.scheduler ?? null, shift: settings.shift ?? null });
}

export function parseModelSettings(text: string | null | undefined): ModelSettings | null {
  if (!text) return null;
  try {
    const value = JSON.parse(text) as Partial<ModelSettings>;
    if (!value || typeof value.files !== 'object' || value.files === null) return null;
    const settings: ModelSettings = { files: value.files as Record<string, string> };
    if (value.sampler != null) settings.sampler = String(value.sampler);
    if (value.scheduler != null) settings.scheduler = String(value.scheduler);
    if (value.shift != null) settings.shift = Number(value.shift);
    return settings;
  } catch {
    return null;
  }
}

/** Whether a stored profile still means what a past record says it was made with. */
export function profileMatchesSettings(profile: Pick<ModelProfile, 'family' | 'files' | 'sampler'>, settings: ModelSettings | null | undefined): boolean {
  return settings !== null && settings !== undefined && serializeModelSettings(profileSettings(profile)) === serializeModelSettings(settings);
}

/** The sampler a family's shipped template uses (what the built-in profile means). */
export function builtInSampler(family: string): SamplerSettings | null {
  return profileFamily(family)?.sampler ?? null;
}
