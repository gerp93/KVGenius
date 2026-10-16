import { useCallback, useState } from 'react';
import type { GenerationRecord } from '../../shared/types';
import { useGenerationChanges } from '../utils/generationChanges';

/**
 * A record shown in a details panel that is not in the page's own list - the original of a result,
 * opened from its panel ("Open original"), which may be filtered out or not loaded. Pages look their
 * list up first and fall back to this one. It follows favorite / pin / hide changes so its panel does
 * not show stale buttons; pages that change a flag themselves call `patch`.
 */
export function useBorrowedRecord() {
  const [borrowed, setBorrowed] = useState<GenerationRecord | null>(null);
  const [expanded, setExpanded] = useState(false);

  const patch = useCallback((id: number, changes: Partial<GenerationRecord>) => {
    setBorrowed((prev) => (prev && prev.id === id ? { ...prev, ...changes } : prev));
  }, []);

  useGenerationChanges((change) => {
    if (change.kind === 'favorite') patch(change.id, { favorite: change.favorite, imagePath: change.imagePath });
    else if (change.kind === 'pinned') patch(change.id, { pinned: change.pinned });
    else if (change.kind === 'hidden') patch(change.id, { hidden: change.hidden });
  });

  return { borrowed, borrow: setBorrowed, patch, expanded, setExpanded };
}
