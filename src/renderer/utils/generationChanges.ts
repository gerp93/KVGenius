import { useEffect, useRef } from 'react';

/** A generation was changed from somewhere that is not the page showing it (the app-wide queue
 * bar), so pages holding it - a Library list, the Generate form - can follow. */
export type GenerationChange =
  /** Favoriting moves the file, so the new path comes along. */
  | { kind: 'favorite'; id: number; favorite: boolean; oldPath: string; imagePath: string }
  | { kind: 'pinned'; id: number; pinned: boolean }
  /** Moved to the Trash: it has left the Library. */
  | { kind: 'trashed'; id: number; imagePath: string };

const EVENT = 'kvgenius:generation-changed';

export function announceGenerationChange(change: GenerationChange): void {
  window.dispatchEvent(new CustomEvent<GenerationChange>(EVENT, { detail: change }));
}

/** Calls `onChange` for every change announced while the calling component is mounted. */
export function useGenerationChanges(onChange: (change: GenerationChange) => void): void {
  const latest = useRef(onChange);
  latest.current = onChange;
  useEffect(() => {
    const listener = (e: Event) => latest.current((e as CustomEvent<GenerationChange>).detail);
    window.addEventListener(EVENT, listener);
    return () => window.removeEventListener(EVENT, listener);
  }, []);
}
