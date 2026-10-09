import { RefObject, useCallback, useLayoutEffect, useRef } from 'react';

/** How long after `hold` the view is kept in place: long enough for the layout changes it was called for (a card leaving, the details panel
 * closing and the grid widening) to have settled. */
const KEEP_MS = 700;

interface Anchor {
  /** Cards from the first one in view onwards (that are not being removed): the first still in the list is the anchor. */
  ids: string[];
  /** How far below the top of the scrolling area the anchor was. */
  offset: number;
  until: number;
}

/**
 * Keeps the Library list where it is when something changes under it - a card is deleted, or the details panel opens or closes and the grid changes
 * width. Without it, closing the panel (which deleting from it does) makes the list shorter, and a list scrolled far down is clamped to its new
 * end, so the page jumps to the bottom.
 *
 * Call `hold(removedIds)` just before the change: it notes which card is at the top of the view (the next one on, if that card is going away) and
 * where. Until the changes have settled, after every render the list is scrolled so that card is back at the same place. Cards are the elements
 * marked `data-card-id` inside `scroller`.
 */
export function useScrollKeeper(scroller: RefObject<HTMLElement | null>): { hold: (removedIds?: number[]) => void } {
  const anchor = useRef<Anchor | null>(null);

  const hold = useCallback(
    (removedIds: number[] = []) => {
      const root = scroller.current;
      if (!root) return;
      const removed = new Set(removedIds.map(String));
      const top = root.getBoundingClientRect().top;
      const cards = [...root.querySelectorAll<HTMLElement>('[data-card-id]')];
      const firstInView = cards.findIndex((card) => card.getBoundingClientRect().bottom > top + 1);
      if (firstInView < 0) return;
      const ids = cards.slice(firstInView, firstInView + 12).map((card) => card.dataset.cardId as string).filter((id) => !removed.has(id));
      if (ids.length === 0) return;
      const first = cards[firstInView];
      // The card to anchor on is the first that stays; its distance from the top is measured from where it is now.
      const anchorCard = cards.find((card) => card.dataset.cardId === ids[0]) ?? first;
      anchor.current = { ids, offset: anchorCard.getBoundingClientRect().top - top, until: Date.now() + KEEP_MS };
    },
    [scroller]
  );

  // After every render while a hold is on: put the anchor card back where it was.
  useLayoutEffect(() => {
    const held = anchor.current;
    const root = scroller.current;
    if (!held || !root) return;
    if (Date.now() > held.until) {
      anchor.current = null;
      return;
    }
    const card = held.ids.map((id) => root.querySelector<HTMLElement>(`[data-card-id="${id}"]`)).find((el) => el !== null);
    if (!card) return;
    const delta = card.getBoundingClientRect().top - root.getBoundingClientRect().top - held.offset;
    if (Math.abs(delta) > 0.5) root.scrollTop += delta;
  });

  return { hold };
}
