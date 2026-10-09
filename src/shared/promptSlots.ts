import { DEFAULT_DENOISE } from './imageToImage';
import type { OutpaintPadding } from './imageToImage';
import type { GenerationKind } from './types';
import type { VideoQuality } from './videoQuality';

/** Everything a prompt "tab" on the Generate page holds - the whole left-hand form, so switching
 * tabs restores exactly what was there. Deliberately plain JSON (no functions/Dates) since this
 * round-trips through app-config.json and an IPC call. */
export interface PromptSlotData {
  mode: GenerationKind;
  prompt: string;
  width: number;
  height: number;
  seed: number;
  seedLocked: boolean;
  steps: number;
  cfg: number;
  lengthSeconds: number;
  /** Video mode only: the Fast (4-step LoRA) / High (20-step) choice. */
  videoQuality: VideoQuality;
  sourceImagePath: string | null;
  /** Video mode only: the Text -> Video / Image -> Video choice. Absent in older configs - image to video (the only kind then). */
  videoFromPicture?: boolean;
  /** Image to video only: the id of the video this run extends (its last frame is the source image and the new clip is joined onto it), or null. Absent in older configs - null. */
  extendFromId?: number | null;
  advancedOpen: boolean;
  customSize: boolean;
  batchSize: number;
  /** Image mode only: the id of the style (Styles tab) added to the prompt when generating, or null for
   * none. Absent in configs saved before styles existed - read as null. A style that has since been
   * deleted is treated as none by the Generate page. */
  styleId?: number | null;
  /** Image mode only: the ids of the elements (Styles tab: reusable parts of a picture, such as an outfit) added to the prompt, in the order picked. Absent in older configs - none. */
  elementIds?: number[];
  /** Image mode only: the id of the saved model (Models page) used instead of the built-in one, or null for the
   * built-in. Absent in configs saved before models existed - read as null. A model that has since been deleted
   * is treated as the built-in by the Generate page. */
  profileId?: number | null;
  /** Image mode only: a picture to start from (image to image), or null for text to image. Kept apart from a
   * video's `sourceImagePath` so switching modes never turns one into the other. Absent in older configs - null. */
  imageSourcePath?: string | null;
  /** Image mode only: the Text -> Image / Image -> Image choice. Absent in older configs - image to image if a start picture was set. */
  imageFromPicture?: boolean;
  /** Image to image only: how much of the start picture is re-drawn (0.05-1). Absent in older configs - the default. */
  denoise?: number;
  /** Image to image only: the painted mask (a kept picture file) that makes it inpainting, or null for none. Absent in older configs - null. */
  maskImagePath?: string | null;
  /** Image to image only: how far the source image is extended on each side (outpainting), or null for none. Absent in older configs - null. */
  outpaint?: OutpaintPadding | null;
  /** The "would repeat the last run" guard - kept per slot so switching away and back doesn't
   * forget it, and it doesn't wrongly carry over between unrelated tabs. */
  lastRunSignature: string | null;
}

export interface PromptSlot {
  id: string;
  /** User-set name; null means "derive one from the prompt text". */
  name: string | null;
  data: PromptSlotData;
}

/** A generous cap, not a real limit anyone should hit - just a guard against an unbounded list. */
export const MAX_PROMPT_SLOTS = 12;

export const MAX_SLOT_NAME_LENGTH = 40;

export function defaultSlotData(mode: GenerationKind = 'image'): PromptSlotData {
  return {
    mode,
    prompt: '',
    width: mode === 'video' ? 640 : 1024,
    height: mode === 'video' ? 640 : 1024,
    seed: Math.floor(Math.random() * 2 ** 32),
    seedLocked: false,
    steps: 8,
    cfg: 1,
    lengthSeconds: 5,
    videoQuality: 'fast',
    sourceImagePath: null,
    videoFromPicture: true,
    extendFromId: null,
    advancedOpen: false,
    customSize: false,
    batchSize: 1,
    styleId: null,
    elementIds: [],
    profileId: null,
    imageSourcePath: null,
    imageFromPicture: false,
    denoise: DEFAULT_DENOISE,
    maskImagePath: null,
    outpaint: null,
    lastRunSignature: null,
  };
}

function randomSlotId(): string {
  return `slot-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function newPromptSlot(mode: GenerationKind = 'image'): PromptSlot {
  return { id: randomSlotId(), name: null, data: defaultSlotData(mode) };
}

/** Most characters of the prompt a tab shows (the rail gives it a few lines; CSS clips the rest). */
const SLOT_LABEL_CHARS = 60;

/** What "derive from the prompt" actually shows for a slot with no explicit name. */
export function slotLabel(slot: Pick<PromptSlot, 'name'>, prompt: string): string {
  if (slot.name) return slot.name;
  const text = prompt.trim();
  if (!text) return 'Untitled';
  return text.length > SLOT_LABEL_CHARS ? `${text.slice(0, SLOT_LABEL_CHARS)}…` : text;
}

function isValidData(data: unknown): data is PromptSlotData {
  if (!data || typeof data !== 'object') return false;
  const d = data as Record<string, unknown>;
  return (
    (d.mode === 'image' || d.mode === 'video') &&
    typeof d.prompt === 'string' &&
    typeof d.width === 'number' &&
    typeof d.height === 'number' &&
    typeof d.seed === 'number' &&
    typeof d.seedLocked === 'boolean' &&
    typeof d.steps === 'number' &&
    typeof d.cfg === 'number' &&
    typeof d.lengthSeconds === 'number' &&
    // Absent in configs saved before the option existed - applySnapshot falls back to 'fast'.
    (d.videoQuality === undefined || d.videoQuality === 'fast' || d.videoQuality === 'high') &&
    (d.sourceImagePath === null || typeof d.sourceImagePath === 'string') &&
    (d.videoFromPicture === undefined || typeof d.videoFromPicture === 'boolean') &&
    (d.extendFromId === undefined || d.extendFromId === null || typeof d.extendFromId === 'number') &&
    typeof d.advancedOpen === 'boolean' &&
    typeof d.customSize === 'boolean' &&
    typeof d.batchSize === 'number' &&
    (d.styleId === undefined || d.styleId === null || typeof d.styleId === 'number') &&
    (d.elementIds === undefined || (Array.isArray(d.elementIds) && d.elementIds.every((id) => typeof id === 'number'))) &&
    (d.profileId === undefined || d.profileId === null || typeof d.profileId === 'number') &&
    (d.imageSourcePath === undefined || d.imageSourcePath === null || typeof d.imageSourcePath === 'string') &&
    (d.imageFromPicture === undefined || typeof d.imageFromPicture === 'boolean') &&
    (d.denoise === undefined || typeof d.denoise === 'number') &&
    (d.maskImagePath === undefined || d.maskImagePath === null || typeof d.maskImagePath === 'string') &&
    (d.outpaint === undefined || d.outpaint === null || typeof d.outpaint === 'object') &&
    (d.lastRunSignature === undefined || d.lastRunSignature === null || typeof d.lastRunSignature === 'string')
  );
}

function isValidSlot(value: unknown): value is PromptSlot {
  if (!value || typeof value !== 'object') return false;
  const s = value as Record<string, unknown>;
  return typeof s.id === 'string' && (s.name === null || typeof s.name === 'string') && isValidData(s.data);
}

/** Defends against a hand-edited or stale config file: coerces to a valid, non-empty slot list. */
export function sanitizeSlots(value: unknown): PromptSlot[] {
  const list = Array.isArray(value) ? value.filter(isValidSlot).slice(0, MAX_PROMPT_SLOTS) : [];
  return list.length > 0 ? list : [newPromptSlot()];
}
