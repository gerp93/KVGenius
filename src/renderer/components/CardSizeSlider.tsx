import { CARD_SCALE, cleanCardScale } from '../hooks/useCardScale';

interface Props {
  scale: number;
  onChange: (scale: number) => void;
}

/** The Library's "how big are the cards" slider: bigger cards, fewer to a row. Double-click it to go back to the usual size. */
export default function CardSizeSlider({ scale, onChange }: Props) {
  return (
    <label className="card-size" title="How big the cards are - bigger cards mean fewer to a row. Double-click to reset.">
      <span className="card-size__icon card-size__icon--small" aria-hidden="true">▫</span>
      <input
        type="range"
        min={CARD_SCALE.min}
        max={CARD_SCALE.max}
        step={CARD_SCALE.step}
        value={scale}
        aria-label="Card size"
        onChange={(e) => onChange(cleanCardScale(Number(e.target.value)))}
        onDoubleClick={() => onChange(CARD_SCALE.default)}
      />
      <span className="card-size__icon card-size__icon--big" aria-hidden="true">▣</span>
    </label>
  );
}
