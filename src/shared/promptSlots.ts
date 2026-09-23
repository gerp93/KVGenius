import type { GenerationKind } from './types';

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
  sourceImagePath: string | null;
  advancedOpen: boolean;
  customSize: boolean;
  batchSize: number;
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
    sourceImagePath: null,
    advancedOpen: false,
    customSize: false,
    batchSize: 1,
    lastRunSignature: null,
  };
}

function randomSlotId(): string {
  return `slot-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function newPromptSlot(mode: GenerationKind = 'image'): PromptSlot {
  return { id: randomSlotId(), name: null, data: defaultSlotData(mode) };
}

/** What "derive from the prompt" actually shows for a slot with no explicit name. */
export function slotLabel(slot: Pick<PromptSlot, 'name'>, prompt: string): string {
  if (slot.name) return slot.name;
  const text = prompt.trim();
  if (!text) return 'Untitled';
  return text.length > 24 ? `${text.slice(0, 24)}…` : text;
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
    (d.sourceImagePath === null || typeof d.sourceImagePath === 'string') &&
    typeof d.advancedOpen === 'boolean' &&
    typeof d.customSize === 'boolean' &&
    typeof d.batchSize === 'number' &&
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
