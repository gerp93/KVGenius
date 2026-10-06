import { useEffect, useRef } from 'react';

/** A generation was favorited or unfavorited from somewhere that is not the page showing it (the
 * app-wide queue panel). Favoriting moves the file, so the new path comes along. */
export interface FavoriteChange {
  id: number;
  favorite: boolean;
  oldPath: string;
  imagePath: string;
}

const EVENT = 'kvgenius:favorite-changed';

export function announceFavoriteChange(change: FavoriteChange): void {
  window.dispatchEvent(new CustomEvent<FavoriteChange>(EVENT, { detail: change }));
}

/** Calls `onChange` for every change announced while the calling component is mounted. */
export function useFavoriteChanges(onChange: (change: FavoriteChange) => void): void {
  const latest = useRef(onChange);
  latest.current = onChange;
  useEffect(() => {
    const listener = (e: Event) => latest.current((e as CustomEvent<FavoriteChange>).detail);
    window.addEventListener(EVENT, listener);
    return () => window.removeEventListener(EVENT, listener);
  }, []);
}
