import { MouseEvent, useEffect, useRef, useState } from 'react';

interface Props {
  /** Copies this text to the clipboard... */
  text?: string;
  /** ...or, instead, the picture in this file (any file the app shows; it lands as an image). */
  imagePath?: string;
  /** Words next to the icon; ignored when `compact`. */
  label?: string;
  /** Icon only (📋 / 🖼️, then ✓), for tight spots like a picture's corner. */
  compact?: boolean;
  title?: string;
  className?: string;
}

const FEEDBACK_MS = 1500;

/** Puts `text` on the clipboard. The async clipboard API first; if the page may not use it, the
 * old select-and-copy route through a throwaway textarea. */
async function writeClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fall through to the older route.
  }
  try {
    const area = document.createElement('textarea');
    area.value = text;
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

/** A small "copy this" button - the prompt text, or a picture - that confirms for a moment once it has copied. */
export default function CopyButton({ text, imagePath, label, compact = false, title, className }: Props) {
  const isImage = imagePath !== undefined;
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [problem, setProblem] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function handleClick(event: MouseEvent) {
    // Often sits on something clickable (a tile that opens the item): copying is all it should do.
    event.stopPropagation();
    let ok: boolean;
    setProblem(null);
    if (isImage) {
      try {
        await window.kvgenius.copyImageToClipboard(imagePath);
        ok = true;
      } catch (err) {
        // Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method".
        const message = err instanceof Error ? err.message : String(err);
        setProblem(message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ''));
        ok = false;
      }
    } else {
      ok = await writeClipboard(text ?? '');
    }
    setState(ok ? 'copied' : 'failed');
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState('idle'), FEEDBACK_MS);
  }

  const icon = isImage ? '🖼️' : '📋';
  const words = label ?? (isImage ? 'Copy image' : 'Copy');
  const idle = compact ? icon : `${icon} ${words}`;
  const shown = state === 'copied' ? (compact ? '✓' : '✓ Copied') : state === 'failed' ? (compact ? '⚠' : 'Copy failed') : idle;
  const baseTitle = title ?? (isImage ? 'Copy the image to the clipboard' : 'Copy the prompt');

  return (
    <button
      type="button"
      className={`copy-button${className ? ` ${className}` : ''}`}
      onClick={handleClick}
      disabled={isImage ? !imagePath : !(text ?? '').trim()}
      title={state === 'failed' ? (problem ?? 'Could not reach the clipboard') : baseTitle}
    >
      {shown}
    </button>
  );
}
