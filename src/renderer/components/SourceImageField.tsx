import ExpandButton from './Lightbox';

interface Props {
  /** The chosen image's file, or null. */
  path: string | null;
  onChooseFile: () => void;
  onChooseFromLibrary: () => void;
}

/**
 * The "Source Image" control shared by video and image to image: two half-width buttons on one line (a file from
 * this computer, or one from the Library), the picture below them and - under it - which file it is. The file name is
 * deliberately not a button, so it is never mistaken for the way to pick a different one.
 */
export default function SourceImageField({ path, onChooseFile, onChooseFromLibrary }: Props) {
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
            <img className="source-image-preview" src={url} alt="Source image" />
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
