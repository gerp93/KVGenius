import { OUTPAINT_MAX_PAD, OUTPAINT_SIDES, OutpaintPadding, normalizeOutpaint, outpaintOutputSize } from '../../shared/imageToImage';

interface Props {
  /** The padding now, or null for none. */
  value: OutpaintPadding | null;
  /** The source picture's size in pixels, once known - the result's size is worked out from it. */
  sourceSize: { width: number; height: number } | null;
  onChange: (value: OutpaintPadding | null) => void;
}

const STEP = 64;
const PRESET = 256;
const SIDE_LABEL = { left: 'Left', top: 'Top', right: 'Right', bottom: 'Bottom' } as const;

/**
 * Outpainting's controls: how many pixels to add beyond each edge of the source picture, laid out around a box that
 * stands for the picture itself. Every side starts at 0; with any side above 0 the run extends the picture instead
 * of re-drawing it (see shared/imageToImage.ts).
 */
export default function ExtendField({ value, sourceSize, onChange }: Props) {
  const pad = value ?? { left: 0, top: 0, right: 0, bottom: 0 };
  const set = (next: Partial<OutpaintPadding>) => onChange(normalizeOutpaint({ ...pad, ...next }));
  const result = value && sourceSize ? outpaintOutputSize(sourceSize.width, sourceSize.height, value) : null;

  return (
    <div className="extend-field">
      <div className="extend-field__head">
        <strong>Extend beyond the frame</strong>
        <span className="settings-hint" style={{ margin: 0 }}>Optional: pixels to add on each side.</span>
      </div>
      <div className="extend-field__grid">
        {OUTPAINT_SIDES.map((side) => (
          <label key={side} className={`extend-field__side extend-field__side--${side}`}>
            <span>{SIDE_LABEL[side]}</span>
            <input
              type="number"
              min={0}
              max={OUTPAINT_MAX_PAD}
              step={STEP}
              value={pad[side]}
              onChange={(e) => set({ [side]: Number(e.target.value) })}
            />
          </label>
        ))}
        <div className="extend-field__picture" aria-hidden="true">
          picture
        </div>
      </div>
      <div className="button-row">
        <button type="button" onClick={() => onChange({ left: PRESET, top: 0, right: PRESET, bottom: 0 })} title="Add 256 px to the left and right">
          Wider
        </button>
        <button type="button" onClick={() => onChange({ left: 0, top: PRESET, right: 0, bottom: PRESET })} title="Add 256 px to the top and bottom">
          Taller
        </button>
        <button type="button" onClick={() => onChange({ left: PRESET, top: PRESET, right: PRESET, bottom: PRESET })} title="Add 256 px on every side">
          All sides
        </button>
        <button type="button" onClick={() => onChange(null)} disabled={value === null}>
          Clear
        </button>
      </div>
      {value && (
        <p className="style-picker__preview" style={{ maxHeight: 'none' }}>
          {result ? `The result is ${result.width} × ${result.height}. ` : ''}The original stays exactly as it is; only the new area is drawn.
          In the prompt, describe what should continue beyond the edges - a prompt describing the whole picture tends to draw the whole picture
          again in the new area. This replaces any painted mask.
        </p>
      )}
    </div>
  );
}
