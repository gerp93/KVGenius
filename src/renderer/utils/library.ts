import { isUpscaleFamily } from '../../shared/upscale';
import { GenerationRecord } from '../../shared/types';

/** An enlarged copy of another picture rather than something generated from a prompt. */
export function isUpscale(record: GenerationRecord): boolean {
  return isUpscaleFamily(record.modelFamily);
}

/** What to tell the user after a pin, given how many pinned items now share the prompt (0 = unpinned). */
export function pinNotice(groupSize: number): string | null {
  if (groupSize <= 0) return null;
  if (groupSize === 1) return 'Pinned - find it under Library > Prompts.';
  const others = groupSize - 1;
  return `Pinned - it joins ${others} other pinned ${others === 1 ? 'picture' : 'pictures'} with this exact prompt, which share one card under Library > Prompts.`;
}
