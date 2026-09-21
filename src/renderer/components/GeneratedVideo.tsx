import { CSSProperties, useEffect, useRef, useState } from 'react';

interface Props {
  src: string;
  /** Local file behind `src`, used to diagnose/repair it and by the "open in default player" fallback. */
  filePath: string;
  /** Muted looping preview for a grid card: no controls, clicks pass through. It plays only while
   * it is on screen, so a long list of videos does not all decode at once. */
  thumbnail?: boolean;
  style?: CSSProperties;
}

const MEDIA_ERROR_LABELS: Record<number, string> = {
  1: 'loading was aborted',
  2: 'a network error occurred',
  3: 'the file could not be decoded',
  4: 'the format is not supported',
};

/**
 * A generated video that says why it failed instead of sitting there as a black box. On an error
 * it also inspects the file once and, if it can fix it (an MP4 whose index is at the end), does
 * and reloads.
 */
export default function GeneratedVideo({ src, filePath, thumbnail, style }: Props) {
  const [error, setError] = useState<string | null>(null);
  const [details, setDetails] = useState<string[] | null>(null);
  const [reloads, setReloads] = useState(0);
  const diagnosedFor = useRef<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    setError(null);
    setDetails(null);
    setReloads(0);
    diagnosedFor.current = null;
  }, [src]);

  // Thumbnails loop while visible and rest on their first frame while scrolled out of view.
  useEffect(() => {
    const el = videoRef.current;
    if (!thumbnail || !el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) el.play().catch(() => undefined);
          else el.pause();
        }
      },
      { rootMargin: '100px' }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [thumbnail, reloads, src]);

  function handleError(e: React.SyntheticEvent<HTMLVideoElement>) {
    const mediaError = e.currentTarget.error;
    const label = mediaError ? (MEDIA_ERROR_LABELS[mediaError.code] ?? 'unknown error') : 'unknown error';
    setError(`${label}${mediaError?.message ? ` (${mediaError.message})` : ''}`);

    if (!filePath || diagnosedFor.current === src) return;
    diagnosedFor.current = src;
    window.kvgenius
      .diagnoseVideo(filePath)
      .then((result) => {
        setDetails(result.lines);
        if (result.repaired) {
          // The file was rewritten: clear the error and load it again (the query string is ignored
          // by the server, it only makes the player treat this as a new resource).
          setError(null);
          setReloads((n) => n + 1);
        }
      })
      .catch(() => undefined);
  }

  const url = `${src}${reloads > 0 ? `?r=${reloads}` : ''}`;
  const video = (
    <video
      ref={videoRef}
      key={url}
      src={thumbnail ? `${url}#t=0.1` : url}
      preload="metadata"
      controls={!thumbnail}
      muted={thumbnail}
      loop={thumbnail}
      playsInline
      style={style}
      onError={handleError}
    />
  );

  if (thumbnail) {
    return (
      <>
        {video}
        {error && (
          <span className="video-error-badge" title={`Cannot preview this video: ${error}`}>
            ⚠
          </span>
        )}
      </>
    );
  }

  return (
    <div className="generated-video">
      {video}
      {error && (
        <div className="generated-video__error">
          <span>Cannot play this video in the app: {error}.</span>
          {details && (
            <details>
              <summary>Technical details</summary>
              <pre>{details.join('\n')}</pre>
            </details>
          )}
          <button type="button" onClick={() => void window.kvgenius.openGenerationExternally(filePath)}>
            Open in default player
          </button>
        </div>
      )}
    </div>
  );
}
