export interface JustifiedItem {
  /** width / height of the item's media. */
  aspect: number;
}

export interface JustifiedRow {
  /** Height every item in this row is drawn at. */
  height: number;
  items: { index: number; width: number }[];
}

// A row closed early (without its last candidate) may stretch above the target height, but not
// by more than this factor - otherwise a lone narrow item would blow up to a huge tile.
const MAX_STRETCH = 1.5;

/**
 * Packs items into rows of equal height that span `containerWidth` exactly, each item keeping
 * its own aspect ratio (a "justified" / Flickr-style layout). Items stay in their given order
 * left-to-right, top-to-bottom. The final, incomplete row keeps `targetHeight` rather than
 * being stretched to fill the width.
 */
export function justifyRows(
  items: JustifiedItem[],
  containerWidth: number,
  targetHeight: number,
  gap: number
): JustifiedRow[] {
  if (containerWidth <= 0) return [];

  const aspects = items.map((item) => Math.max(item.aspect, 0.05));
  const rows: JustifiedRow[] = [];
  let current: number[] = [];
  let aspectSum = 0;

  const heightToFill = (count: number, sum: number) => (containerWidth - gap * (count - 1)) / sum;

  const closeRow = (indexes: number[], height: number) => {
    rows.push({
      height,
      items: indexes.map((index) => ({ index, width: Math.floor(aspects[index] * height) })),
    });
  };

  aspects.forEach((aspect, index) => {
    const withItem = aspectSum + aspect;

    if (withItem * targetHeight + gap * current.length < containerWidth) {
      current.push(index);
      aspectSum = withItem;
      return;
    }

    // Adding this item fills the row. Include it unless closing the row without it lands closer
    // to the target height - that keeps one very wide image from squashing the whole row.
    const heightWith = heightToFill(current.length + 1, withItem);
    if (current.length > 0) {
      const heightWithout = heightToFill(current.length, aspectSum);
      const withoutIsCloser = Math.abs(heightWithout - targetHeight) < Math.abs(heightWith - targetHeight);
      if (withoutIsCloser && heightWithout <= targetHeight * MAX_STRETCH) {
        closeRow(current, heightWithout);
        current = [index];
        aspectSum = aspect;
        return;
      }
    }
    closeRow([...current, index], heightWith);
    current = [];
    aspectSum = 0;
  });

  if (current.length > 0) closeRow(current, targetHeight);
  return rows;
}
