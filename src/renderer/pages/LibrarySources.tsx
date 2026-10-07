import { useEffect, useState } from 'react';
import ExpandButton from '../components/Lightbox';
import type { GenerationQueue } from '../hooks/useGenerationQueue';
import { SOURCE_MISSING_MESSAGE } from '../../shared/sourceFamilies';
import { UPSCALE_FAMILY } from '../../shared/upscale';
import type { SourceImageEntry, VideoSourceRequest } from '../../shared/types';

const FAMILY_LABEL: Record<string, string> = {
  'wan22-i2v': 'video',
  [UPSCALE_FAMILY]: 'upscale',
};

function describeUses(entry: SourceImageEntry): string {
  const kinds = entry.families.map((f) => FAMILY_LABEL[f] ?? f);
  const what = kinds.length > 0 ? ` (${kinds.join(', ')})` : '';
  return `Used by ${entry.uses} result${entry.uses === 1 ? '' : 's'}${what}`;
}

interface Props {
  queue: GenerationQueue;
  /** Send a kept picture to Tools > Upscale. */
  onUpscale: (path: string) => void;
  /** Start a video from a kept picture (opens Generate in video mode with it as the source). */
  onMakeVideo: (request: VideoSourceRequest) => void;
}

/**
 * Library > Sources: every picture that videos and upscales were made from, as the copies the app
 * keeps. They are what Re-rack runs again, and what a result's details panel shows as "Original".
 * Anything uploaded or dropped for a video or an upscale ends up here, so it can be seen and used again.
 */
export default function LibrarySources({ queue, onUpscale, onMakeVideo }: Props) {
  const [entries, setEntries] = useState<SourceImageEntry[] | null>(null);
  const [sizes, setSizes] = useState<Record<string, { width: number; height: number }>>({});
  const [error, setError] = useState<string | null>(null);

  // A finished job may have kept a new source, so load again whenever one finishes.
  const finished = queue.jobs.filter((j) => j.status === 'done').length;
  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listSourceImages()
      .then((list) => {
        if (!cancelled) setEntries(list);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [finished]);

  return (
    <div className="sources-page">
      <h2 className="sources-page__title">Sources</h2>
      <p className="sources-page__hint">
        The pictures your videos and upscales were made from. The app keeps a copy of each, so Re-rack can run them again and a result&apos;s
        details panel can show its original. A copy is removed along with the last result that uses it.
      </p>
      {error && <p className="tools-upscale__error">{error}</p>}
      {entries === null && !error && <p className="sources-page__hint">Loading...</p>}
      {entries?.length === 0 && (
        <p className="sources-page__hint">Nothing here yet. Make a video from a picture, or upscale one, and its original is kept.</p>
      )}
      <div className="sources-page__grid">
        {entries?.map((entry) => {
          const size = sizes[entry.path];
          return (
            <div key={entry.path} className="tools-upscale__card">
              {entry.missing ? (
                <div className="tools-upscale__placeholder" title={SOURCE_MISSING_MESSAGE}>
                  Missing
                </div>
              ) : (
                <div className="sources-page__media">
                  <ExpandButton src={window.kvgenius.imageUrlFor(entry.path)} kind="image" filePath={entry.path} alt="Source picture" />
                  <img
                    src={window.kvgenius.imageUrlFor(entry.path)}
                    alt="Source picture"
                    onLoad={(e) => {
                      const { naturalWidth: width, naturalHeight: height } = e.currentTarget;
                      setSizes((prev) => (prev[entry.path] ? prev : { ...prev, [entry.path]: { width, height } }));
                    }}
                  />
                </div>
              )}
              <div className="tools-upscale__card-text">
                <strong>{size ? `${size.width} × ${size.height}` : entry.missing ? 'Missing' : 'Reading size...'}</strong>
                <span title={entry.missing ? SOURCE_MISSING_MESSAGE : undefined}>
                  {entry.missing ? SOURCE_MISSING_MESSAGE : describeUses(entry)}
                </span>
              </div>
              <div className="tools-upscale__card-actions">
                <button type="button" disabled={entry.missing} onClick={() => onUpscale(entry.path)} title="Upscale this picture again">
                  Upscale
                </button>
                <button
                  type="button"
                  disabled={entry.missing || !size}
                  onClick={() => size && onMakeVideo({ imagePath: entry.path, width: size.width, height: size.height })}
                  title="Make a video from this picture"
                >
                  Make video
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
