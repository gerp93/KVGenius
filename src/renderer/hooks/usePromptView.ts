import { useState } from 'react';
import { SIMILARITY, cleanSimilarity } from '../../shared/promptSimilarity';

const KEY = 'kvgenius-prompts-view';

/** How the Prompts page lays out the pinned prompts. */
export interface PromptView {
  /** Gather prompts that are alike into clusters, each under its own heading. */
  cluster: boolean;
  /** Put the most alike prompts side by side (inside each cluster when clustering). */
  byLikeness: boolean;
  /** How alike (0-1) prompts have to be to share a cluster. */
  threshold: number;
}

const DEFAULTS: PromptView = { cluster: false, byLikeness: false, threshold: SIMILARITY.default };

function load(): PromptView {
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Partial<PromptView> | null;
    return {
      cluster: stored?.cluster === true,
      byLikeness: stored?.byLikeness === true,
      threshold: cleanSimilarity(stored?.threshold ?? DEFAULTS.threshold),
    };
  } catch {
    return DEFAULTS;
  }
}

/** The Prompts page's clustering and ordering choices, remembered between launches. */
export function usePromptView(): [PromptView, (changes: Partial<PromptView>) => void] {
  const [view, setView] = useState(load);
  function update(changes: Partial<PromptView>) {
    const next = { ...view, ...changes, threshold: cleanSimilarity(changes.threshold ?? view.threshold) };
    setView(next);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      // Not remembered - the controls still work.
    }
  }
  return [view, update];
}
