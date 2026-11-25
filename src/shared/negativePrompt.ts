/**
 * Negative prompts: what a picture should avoid. Only the families whose sampler runs a real classifier-free guidance use one - Qwen-Image,
 * HiDream, Ovis-Image and Chroma. Z-Image (CFG 1) and Flux dev (guidance-distilled) have none to give: their graphs zero the negative out,
 * so no field is offered for them.
 *
 * A job's `negativePrompt` is absent for "the template's own" (each shipped template carries a default, below) or the exact text to use
 * (an empty text is a real choice: nothing to avoid). The Library record keeps it the same way (`generations.negative_prompt`, NULL = default).
 */

/** The node every one of these templates holds its negative prompt in (a CLIPTextEncode; main/textToImagePatch.ts sets it). */
export const NEGATIVE_PROMPT_NODE = '7';

/** What each family's shipped template sends as its negative prompt. `negativePrompt.test.ts` checks these against the template files. */
export const DEFAULT_NEGATIVE_PROMPTS: Readonly<Record<string, string>> = {
  'qwen-image': '低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲',
  hidream: 'bad ugly jpeg artifacts',
  'ovis-image': '',
  chroma:
    'This low quality greyscale unfinished sketch is inaccurate and flawed. The image is very blurred and lacks detail with excessive chromatic aberrations and artifacts. The image is overly saturated with excessive bloom. It has a toony aesthetic with bold outlines and flat colors. ',
};

export const MAX_NEGATIVE_PROMPT_LENGTH = 2000;

/** Whether pictures of this family take a negative prompt. */
export function usesNegativePrompt(family: string): boolean {
  return Object.prototype.hasOwnProperty.call(DEFAULT_NEGATIVE_PROMPTS, family);
}

/** The shipped default for a family ('' for one that has none or takes none). */
export function defaultNegativePrompt(family: string): string {
  return usesNegativePrompt(family) ? DEFAULT_NEGATIVE_PROMPTS[family] : '';
}

/**
 * The negative prompt a job should carry: absent (the template's own) unless the user's text differs from the default - so an untouched
 * field changes nothing, and a result made with the default is the same record as one made before this existed. Anything for a family that
 * takes none is dropped.
 */
export function negativePromptForJob(family: string, typed: string | null | undefined): string | undefined {
  if (!usesNegativePrompt(family) || typed === null || typed === undefined) return undefined;
  const text = typed.slice(0, MAX_NEGATIVE_PROMPT_LENGTH);
  return text === DEFAULT_NEGATIVE_PROMPTS[family] ? undefined : text;
}
