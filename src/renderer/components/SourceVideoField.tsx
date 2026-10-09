import GeneratedVideo from './GeneratedVideo';

interface Props {
  /** The chosen video's file, or null. */
  path: string | null;
  onChooseFromLibrary: () => void;
}

/**
 * The "Source Video" control of video to video: a button to pick one of the videos in the Library (only those can be re-drawn - the
 * app reads them where they are), a muted looping preview of the chosen one and which file it is.
 */
export default function SourceVideoField({ path, onChooseFromLibrary }: Props) {
  return (
    <>
      <label className="field-label">Source Video</label>
      <div className="source-image-actions">
        <button type="button" onClick={onChooseFromLibrary} title="Pick one of the videos in your Library">
          🎞️ From Library...
        </button>
      </div>
      {path ? (
        <>
          <div className="source-image-preview-wrap source-video-preview">
            <GeneratedVideo src={window.kvgenius.imageUrlFor(path)} filePath={path} thumbnail />
          </div>
          <div className="source-image-name" title={path}>
            {path.split(/[\\/]/).pop()}
          </div>
        </>
      ) : (
        <p className="source-image-hint">Choose a video from the Library to re-draw.</p>
      )}
    </>
  );
}
