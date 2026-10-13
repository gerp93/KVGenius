import { styleTags } from '../../shared/styles';

/** Small tags naming the styles and elements a picture was made with (from its recorded style label); nothing when it used none.
 * Given `onPick`, each tag is a button (the Library uses it to filter by that style). */
export default function StyleTags({ label, onPick, activeTags }: { label: string | null | undefined; onPick?: (tag: string) => void; activeTags?: string[] }) {
  const tags = styleTags(label);
  if (tags.length === 0) return null;
  const active = new Set((activeTags ?? []).map((t) => t.toLowerCase()));
  return (
    <>
      {tags.map((tag) => {
        const className = `style-tag${active.has(tag.toLowerCase()) ? ' style-tag--on' : ''}`;
        return onPick ? (
          <button
            key={tag}
            type="button"
            className={className}
            title={`Style: ${tag} - click to filter by it`}
            onClick={(e) => {
              e.stopPropagation();
              onPick(tag);
            }}
          >
            {tag}
          </button>
        ) : (
          <span key={tag} className={className} title={`Style: ${tag}`}>
            {tag}
          </span>
        );
      })}
    </>
  );
}
