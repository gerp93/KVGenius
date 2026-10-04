import { MouseEvent, useEffect, useRef, useState } from 'react';

interface Props {
  text: string;
  /** Words next to the icon; ignored when `compact`. */
  label?: string;
  /** Icon only (📋, then ✓), for tight spots like a picture's corner. */
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

/** A small "copy this text" button that confirms for a moment once it has copied. */
export default function CopyButton({ text, label = 'Copy', compact = false, title = 'Copy the prompt', className }: Props) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function handleClick(event: MouseEvent) {
    // Often sits on something clickable (a tile that opens the prompt): copying is all it should do.
    event.stopPropagation();
    const ok = await writeClipboard(text);
    setState(ok ? 'copied' : 'failed');
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState('idle'), FEEDBACK_MS);
  }

  const idle = compact ? '📋' : `📋 ${label}`;
  const shown = state === 'copied' ? (compact ? '✓' : '✓ Copied') : state === 'failed' ? (compact ? '⚠' : 'Copy failed') : idle;

  return (
    <button
      type="button"
      className={`copy-button${className ? ` ${className}` : ''}`}
      onClick={handleClick}
      disabled={!text.trim()}
      title={state === 'failed' ? 'Could not reach the clipboard' : title}
    >
      {shown}
    </button>
  );
}
