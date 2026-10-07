import { GenerationRecord } from '../../shared/types';
import { generationOrigin } from '../../shared/origin';

/** A small tag saying how an item came to be (text to image, image to video, upscaled, GIF); nothing for
 * a kind that is not known. Meant for the badge row of a Library card and for the details panel. */
export default function OriginBadge({ record }: { record: GenerationRecord }) {
  const origin = generationOrigin(record.modelFamily);
  if (!origin) return null;
  return (
    <span className={`library-card__origin-badge library-card__origin-badge--${origin.kind}`} title={origin.title}>
      {origin.label}
    </span>
  );
}
