import { createContext, ReactNode, useContext } from 'react';
import { createPortal } from 'react-dom';

/** Where the app shell docks the details panel: a full-height column at the right of the window,
 * beside the page and the queue bar (so the bar only spans the page, not the panel). */
export const DetailsSlotContext = createContext<HTMLElement | null>(null);

/** Renders its children - a details panel - into that column from wherever it is used (a Library
 * page, or the app shell for a queue result), instead of inside the page's own layout. */
export default function DetailsDock({ children }: { children: ReactNode }) {
  const slot = useContext(DetailsSlotContext);
  return slot ? createPortal(children, slot) : null;
}
