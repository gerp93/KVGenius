import { useEffect, useState } from 'react';
import type { GenerationRecord } from '../../shared/types';
import { needsSourceImage } from '../../shared/sourceFamilies';

/** The kept source image a result needs in order to be re-run, or null if it needs none (or none was
 * ever recorded - results made before sources were kept, which can still be re-run with a new one). */
export function sourcePathOf(record: GenerationRecord | null | undefined): string | null {
  if (!record || !needsSourceImage(record.modelFamily)) return null;
  return record.sourceImagePath ?? null;
}

/**
 * Which of the given source-image paths are no longer on disk. The answer arrives a moment after the
 * paths do; until then nothing counts as missing, so a Re-rack button is not flickered off first.
 */
export function useMissingSources(paths: readonly (string | null | undefined)[]): ReadonlySet<string> {
  const wanted = Array.from(new Set(paths.filter((p): p is string => !!p))).sort();
  const key = JSON.stringify(wanted);
  const [missing, setMissing] = useState<ReadonlySet<string>>(new Set());

  useEffect(() => {
    if (wanted.length === 0) {
      setMissing((prev) => (prev.size === 0 ? prev : new Set()));
      return;
    }
    let cancelled = false;
    window.kvgenius
      .sourceImagesMissing(wanted)
      .then((gone) => {
        if (!cancelled) setMissing(new Set(gone));
      })
      .catch(() => {
        // Could not ask: leave Re-rack on rather than switch it off for a reason nobody can see.
        if (!cancelled) setMissing(new Set());
      });
    return () => {
      cancelled = true;
    };
    // `key` stands for `wanted`, which is a new array every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return missing;
}

/** Whether this one result has lost the source image it was made from. */
export function useSourceMissing(record: GenerationRecord | null | undefined): boolean {
  const path = sourcePathOf(record);
  return useMissingSources([path]).has(path ?? '');
}
