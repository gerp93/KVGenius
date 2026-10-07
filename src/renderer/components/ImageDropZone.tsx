import { useRef, useState } from 'react';
import type { CSSProperties, DragEvent, ReactNode } from 'react';
import { isImageFileName } from '../../shared/imageFiles';
import { isModelFileName } from '../../shared/modelCheck';

interface Props {
  /** The dropped pictures' local paths, ready to use as source images. */
  onPaths: (paths: string[]) => void;
  /** Called with a short message when a drop held nothing usable. */
  onReject?: (message: string) => void;
  /** Take every dropped picture (Tools > Upscale) rather than only the first (a video's source image). */
  multiple?: boolean;
  /** What is accepted: pictures (the default), or model files (.safetensors and friends). */
  kind?: 'image' | 'model';
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}

function carriesFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer.types).includes('Files');
}

/**
 * A region that accepts picture files dragged in from the file manager. Only the area it wraps takes
 * a drop; it outlines itself while a file is held over it. Dropped files become paths through the
 * preload (the renderer cannot read a dropped file's path itself), which also allows the app to show
 * and upload them, as a file dialog pick does.
 */
export default function ImageDropZone({ onPaths, onReject, multiple = false, kind = 'image', className, style, children }: Props) {
  const [over, setOver] = useState(false);
  // dragenter / dragleave fire for every child entered or left; count them to know when the drag really left.
  const depth = useRef(0);

  function handleEnter(event: DragEvent) {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth.current += 1;
    setOver(true);
  }

  function handleOver(event: DragEvent) {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  }

  function handleLeave(event: DragEvent) {
    if (!carriesFiles(event)) return;
    depth.current = Math.max(0, depth.current - 1);
    if (depth.current === 0) setOver(false);
  }

  async function handleDrop(event: DragEvent) {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth.current = 0;
    setOver(false);
    const accepted = kind === 'model' ? isModelFileName : isImageFileName;
    const images = Array.from(event.dataTransfer.files).filter((file) => accepted(file.name));
    if (images.length === 0) {
      onReject?.(kind === 'model' ? 'Drop a model file (.safetensors).' : 'Drop PNG, JPG or WebP pictures.');
      return;
    }
    try {
      const chosen = multiple ? images : images.slice(0, 1);
      const paths = await (kind === 'model' ? window.kvgenius.droppedModelFilePaths(chosen) : window.kvgenius.droppedImagePaths(chosen));
      if (paths.length === 0) onReject?.('Those files could not be used.');
      else onPaths(paths);
    } catch (err) {
      onReject?.(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div
      className={`image-drop${over ? ' image-drop--over' : ''}${className ? ` ${className}` : ''}`}
      style={style}
      onDragEnter={handleEnter}
      onDragOver={handleOver}
      onDragLeave={handleLeave}
      onDrop={(event) => void handleDrop(event)}
    >
      {children}
    </div>
  );
}
