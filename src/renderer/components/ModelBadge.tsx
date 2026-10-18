import { GenerationRecord } from '../../shared/types';
import { modelLabel } from '../../shared/modelLabel';

/** A small tag naming the model an item was made with (its saved variant, else the family's built-in model);
 * nothing for an item no generation model made. Sits beside the OriginBadge. */
export default function ModelBadge({ record }: { record: GenerationRecord }) {
  const label = modelLabel(record);
  if (!label) return null;
  return (
    <span className="library-card__model-badge" title={`Made with ${label}`}>
      {label}
    </span>
  );
}
