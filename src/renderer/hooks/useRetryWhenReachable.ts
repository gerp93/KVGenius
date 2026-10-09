import { useEffect, useRef } from 'react';

const POLL_MS = 3000;

/**
 * While `waiting` is true (something is showing "ComfyUI is not reachable"), checks every few seconds and calls `retry` as soon as ComfyUI answers,
 * so what was waiting on it fills in by itself once it is started - nobody has to find the "try again" link.
 */
export function useRetryWhenReachable(waiting: boolean, retry: () => void): void {
  const latest = useRef(retry);
  latest.current = retry;
  useEffect(() => {
    if (!waiting) return;
    let cancelled = false;
    const interval = setInterval(() => {
      window.kvgenius
        .checkComfyUIConnection()
        .then((reachable) => {
          if (reachable && !cancelled) latest.current();
        })
        .catch(() => undefined);
    }, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [waiting]);
}
