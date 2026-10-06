import { useEffect } from 'react';
import GeneratedVideo from './GeneratedVideo';

interface Props {
  /** The files to show; `index` picks which one is on screen. */
  paths: string[];
  index: number;
  isVideo: boolean;
  alt: string;
}

/**
 * The picture (or video thumbnail) of a card that stands for several items: shows `paths[index]` and
 * fades each new one in. Sized by its card's media box, like a plain <img>.
 */
export default function CycleMedia({ paths, index, isVideo, alt }: Props) {
  const path = paths[index] ?? paths[0];
  const next = paths.length > 1 ? paths[(index + 1) % paths.length] : null;

  // Have the next picture ready, so the swap does not flash an empty frame.
  useEffect(() => {
    if (!next || isVideo) return;
    const preload = new Image();
    preload.src = window.kvgenius.imageUrlFor(next);
  }, [next, isVideo]);

  const url = window.kvgenius.imageUrlFor(path);
  return isVideo ? (
    <GeneratedVideo key={path} src={url} filePath={path} thumbnail />
  ) : (
    <img key={path} className={paths.length > 1 ? 'cycle-fade' : undefined} src={url} alt={alt} decoding="async" />
  );
}
