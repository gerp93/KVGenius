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
 * Outpainting's controls: how many pixels to add beyond each edge of the source picture, as one row of four numbers
 * and a row of presets. Every side starts at 0; with any side above 0 the run extends the picture instead of
 * re-drawing it (see shared/imageToImage.ts). Generate shows it folded away until the user asks for it.
 */
export default function ExtendField({ value, sourceSize, onChange }: Props) {
  const pad = value ?? { left: 0, top: 0, right: 0, bottom: 0 };
  const set = (next: Partial<OutpaintPadding>) => onChange(normalizeOutpaint({ ...pad, ...next }));
  const result = value && sourceSize ? outpaintOutputSize(sourceSize.width, sourceSize.height, value) : null;

  return (
    <div className="extend-field">
      <div className="extend-field__sides">
        {OUTPAINT_SIDES.map((side) => (
          <label key={side} className="extend-field__side">
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
      </div>
      <div className="extend-field__presets">
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
        <p className="source-image-hint">
          {result ? `The result is ${result.width} × ${result.height}. ` : ''}The original stays exactly as it is; only the new area is drawn.
          Describe what should continue beyond the edges, not the whole picture. Replaces any painted mask.
        </p>
      )}
    </div>
  );
}
