import { useEffect, useState } from 'react';
import GeneratedVideo from './GeneratedVideo';

interface Props {
  /** The files to show; `index` picks which one is on screen. */
  paths: string[];
  index: number;
  isVideo: boolean;
  alt: string;
}

/** How long a new picture takes to fade in over the one before it (keep in step with `.cycle-fade` in index.css). */
export const CYCLE_FADE_MS = 1400;

/**
 * The picture (or video thumbnail) of a card that stands for several items: shows `paths[index]` and
 * crossfades to each new one - the previous picture stays underneath while the next fades in over it,
 * so the card never dips to its background between them. Sized by its card's media box, like a plain <img>.
 */
export default function CycleMedia({ paths, index, isVideo, alt }: Props) {
  const path = paths[index] ?? paths[0];
  const next = paths.length > 1 ? paths[(index + 1) % paths.length] : null;

  // The picture on top, and - only while it is still fading in - the one it is replacing.
  const [shown, setShown] = useState(path);
  const [under, setUnder] = useState<string | null>(null);
  if (shown !== path) {
    setUnder(shown);
    setShown(path);
  }
  useEffect(() => {
    if (under === null) return;
    const done = window.setTimeout(() => setUnder(null), CYCLE_FADE_MS + 100);
    return () => window.clearTimeout(done);
  }, [under]);

  // Have the next picture ready, so the swap does not wait on the file.
  useEffect(() => {
    if (!next || isVideo) return;
    const preload = new Image();
    preload.src = window.kvgenius.imageUrlFor(next);
  }, [next, isVideo]);

  function layer(file: string, className?: string) {
    const url = window.kvgenius.imageUrlFor(file);
    return isVideo ? (
      <span key={file} className={className}>
        <GeneratedVideo src={url} filePath={file} thumbnail />
      </span>
    ) : (
      <img key={file} className={className} src={url} alt={alt} decoding="async" />
    );
  }

  return (
    <>
      {under !== null && layer(under, 'cycle-under')}
      {layer(shown, under !== null ? 'cycle-fade' : undefined)}
    </>
  );
}
