import { useEffect, useState } from 'react';
import { BackgroundWork, NO_BACKGROUND_WORK } from '../../shared/backgroundWork';

/** What else is using the queue that the queue panel does not list: the Styles page's examples and outside clients' requests. */
export function useBackgroundWork(): BackgroundWork {
  const [work, setWork] = useState<BackgroundWork>(NO_BACKGROUND_WORK);
  useEffect(() => {
    let cancelled = false;
    const unsubscribe = window.kvgenius.onBackgroundWork(setWork);
    window.kvgenius
      .getBackgroundWork()
      .then((current) => {
        if (!cancelled) setWork(current);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);
  return work;
}
