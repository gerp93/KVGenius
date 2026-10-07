import { useCallback, useEffect, useState } from 'react';
import { ModelStatusReport } from '../../shared/modelStatus';

const POLL_MS = 5000;

/** Which model files are available (see getModelStatus), rechecked every few seconds so the page
 * notices by itself when the user copies a file in or starts ComfyUI. `refresh` rechecks right now. */
export function useModelStatus(): { report: ModelStatusReport | null; refresh: () => void } {
  const [report, setReport] = useState<ModelStatusReport | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const next = await window.kvgenius.getModelStatus();
        if (!cancelled) setReport(next);
      } catch (err) {
        console.error('Could not read the model status', err);
      }
    }
    void check();
    const interval = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [tick]);

  const refresh = useCallback(() => setTick((n) => n + 1), []);
  return { report, refresh };
}
