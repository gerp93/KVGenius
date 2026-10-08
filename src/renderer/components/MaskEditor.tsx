import { PointerEvent, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import './MaskEditor.css';

interface Props {
  /** The picture the mask is painted on (any URL the app can show). */
  imageUrl: string;
  /** A mask painted earlier, as a PNG data URL (white = re-draw), to carry on from; null to start empty. */
  existingMask: string | null;
  /** The finished mask as a black and white PNG data URL (white = re-draw), or null when nothing was painted. */
  onDone: (maskDataUrl: string | null) => void;
  onCancel: () => void;
}

/** The painting canvas is no larger than this on its long side: the mask only needs to be as sharp as the output. */
const MAX_SIDE = 1536;
const UNDO_LIMIT = 10;
const PAINT = 'rgb(255, 64, 64)';

type Tool = 'brush' | 'eraser';

/** True when anything is painted (any pixel with some opacity). */
function hasPaint(ctx: CanvasRenderingContext2D, width: number, height: number): boolean {
  const data = ctx.getImageData(0, 0, width, height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
  return false;
}

/**
 * A full-screen editor for the mask of an inpainting run: paint over the picture where it should be re-drawn, with a
 * brush and an eraser, and undo / clear / invert. The painting is a red tint over the picture; what it hands back is a
 * black and white picture at the same shape (white = re-draw), which is what the workflow takes.
 */
export default function MaskEditor({ imageUrl, existingMask, onDone, onCancel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const undoStack = useRef<ImageData[]>([]);
  const drawing = useRef<{ x: number; y: number } | null>(null);
  const [tool, setTool] = useState<Tool>('brush');
  // The brush's width as a percentage of the picture's shorter side.
  const [size, setSize] = useState(6);
  const [ready, setReady] = useState(false);
  const [undoCount, setUndoCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const ctx = () => canvasRef.current?.getContext('2d', { willReadFrequently: true }) ?? null;

  function snapshot() {
    const c = canvasRef.current;
    const g = ctx();
    if (!c || !g) return;
    undoStack.current.push(g.getImageData(0, 0, c.width, c.height));
    if (undoStack.current.length > UNDO_LIMIT) undoStack.current.shift();
    setUndoCount(undoStack.current.length);
  }

  /** Sizes the canvas to the picture and, if a mask was painted before, paints it back in. */
  const handleImageLoad = useCallback(
    (img: HTMLImageElement) => {
      const c = canvasRef.current;
      if (!c) return;
      const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
      c.width = Math.max(1, Math.round(img.naturalWidth * scale));
      c.height = Math.max(1, Math.round(img.naturalHeight * scale));
      if (!existingMask) {
        setReady(true);
        return;
      }
      const old = new Image();
      old.onload = () => {
        const tmp = document.createElement('canvas');
        tmp.width = c.width;
        tmp.height = c.height;
        const t = tmp.getContext('2d', { willReadFrequently: true });
        const g = c.getContext('2d', { willReadFrequently: true });
        if (t && g) {
          t.drawImage(old, 0, 0, c.width, c.height);
          const px = t.getImageData(0, 0, c.width, c.height);
          // White (the red channel) becomes paint of that opacity; black becomes nothing.
          for (let i = 0; i < px.data.length; i += 4) {
            const amount = px.data[i];
            px.data[i] = 255;
            px.data[i + 1] = 64;
            px.data[i + 2] = 64;
            px.data[i + 3] = amount;
          }
          g.putImageData(px, 0, 0);
        }
        setReady(true);
      };
      old.onerror = () => setReady(true);
      old.src = existingMask;
    },
    [existingMask],
  );

  function point(e: PointerEvent<HTMLCanvasElement>) {
    const c = e.currentTarget;
    const rect = c.getBoundingClientRect();
    return { x: ((e.clientX - rect.left) * c.width) / rect.width, y: ((e.clientY - rect.top) * c.height) / rect.height };
  }

  function stroke(from: { x: number; y: number }, to: { x: number; y: number }) {
    const c = canvasRef.current;
    const g = ctx();
    if (!c || !g) return;
    g.save();
    g.globalCompositeOperation = tool === 'brush' ? 'source-over' : 'destination-out';
    g.strokeStyle = PAINT;
    g.fillStyle = PAINT;
    g.lineCap = 'round';
    g.lineJoin = 'round';
    g.lineWidth = Math.max(1, (Math.min(c.width, c.height) * size) / 100);
    g.beginPath();
    g.moveTo(from.x, from.y);
    g.lineTo(to.x, to.y);
    g.stroke();
    g.restore();
  }

  function handleDown(e: PointerEvent<HTMLCanvasElement>) {
    if (!ready) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    snapshot();
    const p = point(e);
    drawing.current = p;
    stroke(p, p);
  }

  function handleMove(e: PointerEvent<HTMLCanvasElement>) {
    if (!drawing.current) return;
    const p = point(e);
    stroke(drawing.current, p);
    drawing.current = p;
  }

  function handleUp() {
    drawing.current = null;
  }

  const undo = useCallback(() => {
    const c = canvasRef.current;
    const g = ctx();
    const last = undoStack.current.pop();
    if (!c || !g || !last) return;
    g.putImageData(last, 0, 0);
    setUndoCount(undoStack.current.length);
  }, []);

  function clear() {
    const c = canvasRef.current;
    const g = ctx();
    if (!c || !g) return;
    snapshot();
    g.clearRect(0, 0, c.width, c.height);
  }

  /** Paints what is empty and unpaints what is painted. */
  function invert() {
    const c = canvasRef.current;
    const g = ctx();
    if (!c || !g) return;
    snapshot();
    const px = g.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < px.data.length; i += 4) {
      px.data[i] = 255;
      px.data[i + 1] = 64;
      px.data[i + 2] = 64;
      px.data[i + 3] = 255 - px.data[i + 3];
    }
    g.putImageData(px, 0, 0);
  }

  function done() {
    const c = canvasRef.current;
    const g = ctx();
    if (!c || !g) return;
    if (!hasPaint(g, c.width, c.height)) {
      onDone(null);
      return;
    }
    try {
      const out = document.createElement('canvas');
      out.width = c.width;
      out.height = c.height;
      const o = out.getContext('2d');
      if (!o) throw new Error('no canvas');
      const px = g.getImageData(0, 0, c.width, c.height);
      // Black where nothing is painted, white where it is (a partly painted edge pixel is a grey in between).
      for (let i = 0; i < px.data.length; i += 4) {
        const v = px.data[i + 3];
        px.data[i] = v;
        px.data[i + 1] = v;
        px.data[i + 2] = v;
        px.data[i + 3] = 255;
      }
      o.putImageData(px, 0, 0);
      onDone(out.toDataURL('image/png'));
    } catch {
      setError('The mask could not be read from the canvas.');
    }
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onCancel();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undo();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel, undo]);

  return createPortal(
    <div className="mask-editor" role="dialog" aria-label="Paint a mask">
      <div className="mask-editor__bar">
        <strong>Paint where it should change</strong>
        <div className="mask-editor__tools" role="radiogroup" aria-label="Tool">
          <button type="button" role="radio" aria-checked={tool === 'brush'} className={tool === 'brush' ? 'primary' : undefined} onClick={() => setTool('brush')}>
            🖌️ Brush
          </button>
          <button type="button" role="radio" aria-checked={tool === 'eraser'} className={tool === 'eraser' ? 'primary' : undefined} onClick={() => setTool('eraser')}>
            🧽 Eraser
          </button>
        </div>
        <label className="mask-editor__size">
          Size
          <input type="range" min={1} max={30} value={size} onChange={(e) => setSize(Number(e.target.value))} />
        </label>
        <button type="button" onClick={undo} disabled={undoCount === 0} title="Undo the last stroke (Ctrl+Z)">
          ↶ Undo
        </button>
        <button type="button" onClick={clear}>
          Clear
        </button>
        <button type="button" onClick={invert} title="Swap the painted and unpainted areas">
          Invert
        </button>
        <span className="mask-editor__spacer" />
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="primary" onClick={done} disabled={!ready}>
          Done
        </button>
      </div>
      {error && <p className="mask-editor__error">{error}</p>}
      <div className="mask-editor__stage">
        <div className="mask-editor__frame">
          <img src={imageUrl} alt="The picture to paint on" draggable={false} onLoad={(e) => handleImageLoad(e.currentTarget)} onError={() => setError('The picture could not be loaded.')} />
          <canvas
            ref={canvasRef}
            className="mask-editor__canvas"
            onPointerDown={handleDown}
            onPointerMove={handleMove}
            onPointerUp={handleUp}
            onPointerCancel={handleUp}
          />
        </div>
      </div>
      <p className="mask-editor__hint">Red is re-drawn; everything else stays as it is. Leave it empty and Done to use no mask.</p>
    </div>,
    document.body,
  );
}
