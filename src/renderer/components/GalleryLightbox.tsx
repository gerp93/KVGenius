import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import GeneratedVideo from './GeneratedVideo';

interface Props {
  src: string;
  kind: 'image' | 'video';
  filePath: string;
  alt?: string;
  hasPrev: boolean;
  hasNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}

/**
 * Full-window viewer for browsing a whole gallery in order (e.g. the Library Output grid), not
 * just one image: arrow buttons plus the Left/Right/Escape keys move through the same list the
 * grid shows, in the same order. See Lightbox.tsx for the single-item version this mirrors.
 */
export default function GalleryLightbox({ src, kind, filePath, alt, hasPrev, hasNext, onPrev, onNext, onClose }: Props) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft' && hasPrev) onPrev();
      else if (e.key === 'ArrowRight' && hasNext) onNext();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, onPrev, onNext, hasPrev, hasNext]);

  return createPortal(
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        e.stopPropagation();
        onClose();
      }}
    >
      <button type="button" className="lightbox__close" title="Close (Esc)" onClick={onClose}>
        ✕
      </button>
      {hasPrev && (
        <button
          type="button"
          className="lightbox__nav lightbox__nav--prev"
          title="Previous (←)"
          onClick={(e) => {
            e.stopPropagation();
            onPrev();
          }}
        >
          ‹
        </button>
      )}
      {hasNext && (
        <button
          type="button"
          className="lightbox__nav lightbox__nav--next"
          title="Next (→)"
          onClick={(e) => {
            e.stopPropagation();
            onNext();
          }}
        >
          ›
        </button>
      )}
      <div className="lightbox__media" onClick={(e) => e.stopPropagation()}>
        {kind === 'video' ? <GeneratedVideo src={src} filePath={filePath} /> : <img src={src} alt={alt ?? ''} />}
      </div>
    </div>,
    document.body
  );
}
