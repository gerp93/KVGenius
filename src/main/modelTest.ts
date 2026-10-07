import type { ModelTestResult } from '../shared/modelCheck';
import { GenerationParams } from '../shared/types';
import { profileFamily } from '../shared/modelFamilies';
import { ModelProfileInput, profileSettings, validateProfileInput } from '../shared/modelProfiles';

/** Steps a test render is capped at: it only has to prove the files load and run, so it stays quick. */
const TEST_MAX_STEPS = 8;
const TEST_SIZE = 256;
const TEST_PROMPT = 'a red apple on a wooden table, soft daylight, photo';

/**
 * Turns what ComfyUI or the network said into something a person can act on. The raw text is kept at the end for
 * anything not recognised, so nothing is hidden.
 */
export function explainTestFailure(raw: string): string {
  const text = raw.replace(/^Error invoking remote method '[^']+': (Error: )?/, '').trim();
  if (/not reachable|ECONNREFUSED|fetch failed/i.test(text)) return 'ComfyUI is not reachable. Start it and try again.';
  if (/value_not_in_list|not in \[|Value not in list/i.test(text)) {
    return 'ComfyUI does not list one of the chosen files. Put it in the right folder, restart ComfyUI (or press R in its window) and use Check Again.';
  }
  if (/out of memory|OOM|allocate/i.test(text)) return 'ComfyUI ran out of memory. Close other programs that use the graphics card, or pick a smaller model.';
  if (/size mismatch|shape|mat1 and mat2|state_dict|dimension|invalid for input|Missing key|Unexpected key|incompatible/i.test(text)) {
    return `The chosen files do not fit together - they are probably from different kinds of models. (${text.slice(0, 200)})`;
  }
  return text.length > 300 ? `${text.slice(0, 300)}...` : text || 'The test did not work, and ComfyUI gave no reason.';
}

export interface ModelTestDeps {
  /** Runs `work` only while the job queue is idle (see JobQueue.runExclusive). */
  runExclusive: <T>(work: () => Promise<T>) => Promise<T>;
  generate: (family: string, params: GenerationParams) => Promise<{ bytes: Buffer; extension: string }>;
}

/**
 * Renders one small picture with a model as it is set up in the editor (saved or not), to prove it loads and runs.
 * It does not touch the Library or the queue. The picture is a few steps at 256 px, so what it looks like says
 * little about the quality of the real settings - only that the files work together.
 */
export async function runModelTest(input: ModelProfileInput, deps: ModelTestDeps): Promise<ModelTestResult> {
  // The name is not what is being tested, so an empty one must not stop the test.
  const checked = validateProfileInput({ ...input, name: input.name?.trim() || 'test' });
  if (!checked.ok) return { ok: false, message: checked.message };
  const family = profileFamily(checked.value.family);
  if (!family) return { ok: false, message: 'This kind of model cannot be tested.' };

  const steps = Math.min(checked.value.sampler.steps, TEST_MAX_STEPS);
  const params: GenerationParams = {
    prompt: TEST_PROMPT,
    width: TEST_SIZE,
    height: TEST_SIZE,
    seed: 1234,
    steps,
    cfg: checked.value.sampler.cfg,
    modelName: checked.value.name,
    modelSettings: profileSettings(checked.value),
  };
  const started = Date.now();
  try {
    const output = await deps.runExclusive(() => deps.generate(checked.value.family, params));
    const seconds = Math.max(1, Math.round((Date.now() - started) / 1000));
    const mime = output.extension.toLowerCase() === '.png' ? 'image/png' : output.extension.toLowerCase() === '.webp' ? 'image/webp' : 'image/jpeg';
    return {
      ok: true,
      message: `It loaded and rendered a ${TEST_SIZE}x${TEST_SIZE} test picture in ${seconds} s, at ${steps} step${steps === 1 ? '' : 's'}. The files work together; judge the settings on a full-size picture.`,
      imageBase64: output.bytes.toString('base64'),
      mime,
    };
  } catch (err) {
    return { ok: false, message: explainTestFailure(err instanceof Error ? err.message : String(err)) };
  }
}
