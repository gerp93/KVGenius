import ExpandButton from './Lightbox';
import type { OutpaintPadding } from '../../shared/imageToImage';

interface Props {
  /** The chosen image's file, or null. */
  path: string | null;
  onChooseFile: () => void;
  onChooseFromLibrary: () => void;
  /** Image -> Image extending the picture beyond its edges: the picture's size and the padding, to draw the new canvas around it. */
  extend?: { width: number; height: number; pad: OutpaintPadding } | null;
}

/**
 * The "Source Image" control shared by video and image to image: two half-width buttons on one line (a file from
 * this computer, or one from the Library), the picture below them and - under it - which file it is. The file name is
 * deliberately not a button, so it is never mistaken for the way to pick a different one. While the picture is being
 * extended, the preview is the finished canvas: the picture sits where it will end up and the hatched blank around it
 * is the area that will be drawn.
 */
export default function SourceImageField({ path, onChooseFile, onChooseFromLibrary, extend }: Props) {
  const url = path ? window.kvgenius.imageUrlFor(path) : null;
  return (
    <>
      <label className="field-label">Source Image</label>
      <div className="source-image-actions">
        <button type="button" onClick={onChooseFile} title="Pick an image file from this computer (or drop one here)">
          📁 Choose file...
        </button>
        <button type="button" onClick={onChooseFromLibrary} title="Pick one of the images in your Library">
          🖼️ From Library...
        </button>
      </div>
      {path && url ? (
        <>
          <div className="source-image-preview-wrap">
            <ExpandButton src={url} kind="image" filePath={path} alt="Source image" />
            {extend ? <ExtendedPreview url={url} extend={extend} /> : <img className="source-image-preview" src={url} alt="Source image" />}
          </div>
          <div className="source-image-name" title={path}>
            {path.split(/[\\/]/).pop()}
          </div>
        </>
      ) : (
        <p className="source-image-hint">or drop an image here</p>
      )}
    </>
  );
}

function ExtendedPreview({ url, extend }: { url: string; extend: NonNullable<Props['extend']> }) {
  const { width, height, pad } = extend;
  const totalWidth = width + pad.left + pad.right;
  const totalHeight = height + pad.top + pad.bottom;
  const pct = (n: number, of: number) => `${(n / of) * 100}%`;
  return (
    <div
      className="source-image-canvas"
      style={{ ['--ratio' as string]: totalWidth / totalHeight, aspectRatio: `${totalWidth} / ${totalHeight}` }}
      title={`The picture stays as it is; the hatched area is drawn new (${totalWidth} × ${totalHeight} before it is scaled to fit)`}
    >
      <img
        src={url}
        alt="Source image, where it sits in the extended picture"
        style={{ left: pct(pad.left, totalWidth), top: pct(pad.top, totalHeight), width: pct(width, totalWidth), height: pct(height, totalHeight) }}
      />
    </div>
  );
}
