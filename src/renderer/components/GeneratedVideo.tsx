import { CSSProperties, useEffect, useState } from 'react';

interface Props {
  src: string;
  /** Local file behind `src`, used by the "open in default player" fallback. */
  filePath: string;
  /** Muted first-frame preview for a grid card: no controls, clicks pass through. */
  thumbnail?: boolean;
  style?: CSSProperties;
}

const MEDIA_ERROR_LABELS: Record<number, string> = {
  1: 'loading was aborted',
  2: 'a network error occurred',
  3: 'the file could not be decoded',
  4: 'the format is not supported',
};

/** A generated video that says why it failed instead of sitting there as a black box. */
export default function GeneratedVideo({ src, filePath, thumbnail, style }: Props) {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setError(null), [src]);

  const video = (
    <video
      key={src}
      src={thumbnail ? `${src}#t=0.1` : src}
      preload="metadata"
      controls={!thumbnail}
      muted={thumbnail}
      playsInline
      style={style}
      onError={(e) => {
        const mediaError = e.currentTarget.error;
        const label = mediaError ? (MEDIA_ERROR_LABELS[mediaError.code] ?? 'unknown error') : 'unknown error';
        setError(`${label}${mediaError?.message ? ` (${mediaError.message})` : ''}`);
      }}
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
          <button type="button" onClick={() => void window.kvgenius.openGenerationExternally(filePath)}>
            Open in default player
          </button>
        </div>
      )}
    </div>
  );
}
