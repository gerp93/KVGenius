import { useEffect, useState } from 'react';

const DEFAULT_INTERVAL_MS = 2500;

function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

/**
 * Index of the picture a card should show, stepping through `count` pictures on its own. It rests while
 * `paused` (so what is under the pointer is what a click opens), with a single picture, and for people
 * who have asked their system for less motion. The first step comes after a random head start, so a
 * screenful of stacks does not all flip at the same instant.
 */
export function useCycleIndex(count: number, paused = false, intervalMs = DEFAULT_INTERVAL_MS): number {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (count < 2 || paused || prefersReducedMotion()) return;
    let interval: number | undefined;
    const head = window.setTimeout(() => {
      setIndex((i) => (i + 1) % count);
      interval = window.setInterval(() => setIndex((i) => (i + 1) % count), intervalMs);
    }, intervalMs * (0.4 + Math.random() * 0.6));
    return () => {
      window.clearTimeout(head);
      window.clearInterval(interval);
    };
  }, [count, paused, intervalMs]);

  // The list can shrink under us (an item was unpinned or deleted).
  return count > 0 ? index % count : 0;
}
