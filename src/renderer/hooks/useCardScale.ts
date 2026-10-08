import { useState } from 'react';

const KEY = 'kvgenius-library-card-scale';

/** How much bigger or smaller than usual the Library's cards are drawn. 1 is the usual size. */
export const CARD_SCALE = { min: 0.75, max: 2, step: 0.05, default: 1 } as const;

/** A scale from storage or a slider: a number within the limits, else the usual size. */
export function cleanCardScale(value: unknown): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return CARD_SCALE.default;
  return Math.min(CARD_SCALE.max, Math.max(CARD_SCALE.min, Math.round(n / CARD_SCALE.step) * CARD_SCALE.step));
}

function load(): number {
  try {
    const stored = localStorage.getItem(KEY);
    return stored === null ? CARD_SCALE.default : cleanCardScale(stored);
  } catch {
    return CARD_SCALE.default;
  }
}

/** The Library's card size, one setting for Output, Prompts and the Trash, remembered between launches. */
export function useCardScale(): [number, (scale: number) => void] {
  const [scale, setScaleState] = useState(load);
  function setScale(next: number) {
    const clean = cleanCardScale(next);
    setScaleState(clean);
    try {
      localStorage.setItem(KEY, String(clean));
    } catch {
      // Not remembered - the slider still works.
    }
  }
  return [scale, setScale];
}
