import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import GeneratedVideo from './GeneratedVideo';

interface Props {
  src: string;
  kind: 'image' | 'video';
  /** Local file behind `src` (used by the video player's fallback). */
  filePath?: string;
  alt?: string;
}

/**
 * A small expand button for the top-left corner of whatever shows an image or video (the parent
 * needs `position: relative`). Opens the media as large as fits in a full-window overlay, with a
 * small margin all round. Click outside, the ✕ button or Esc closes it.
 */
export default function ExpandButton({ src, kind, filePath, alt }: Props) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open]);

  return (
    <>
      <button
        type="button"
        className="expand-button"
        title="Expand"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
      >
        ⤢
      </button>
      {open &&
        createPortal(
          // React events bubble through portals to the parent component tree, so stop the click here
          // or it would also hit whatever card/panel the button sits in.
          <div
            className="lightbox"
            role="dialog"
            aria-modal="true"
            onClick={(e) => {
              e.stopPropagation();
              setOpen(false);
            }}
          >
            <button type="button" className="lightbox__close" title="Close" onClick={() => setOpen(false)}>
              ✕
            </button>
            <div className="lightbox__media" onClick={(e) => e.stopPropagation()}>
              {kind === 'video' ? (
                <GeneratedVideo src={src} filePath={filePath ?? ''} />
              ) : (
                <img src={src} alt={alt ?? ''} />
              )}
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
