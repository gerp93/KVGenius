/**
 * Picks the column count that makes `count` equally-shaped tiles (width / height = `aspect`) as
 * large as possible while all fitting inside `width` x `height` - so a handful of results fill
 * the viewer, and a lot of them shrink to still fit without scrolling.
 */
export function fitGrid(
  count: number,
  aspect: number,
  width: number,
  height: number,
  gap: number
): { cols: number; tileWidth: number; tileHeight: number } {
  if (count <= 0 || width <= 0 || height <= 0 || aspect <= 0) return { cols: 1, tileWidth: 0, tileHeight: 0 };

  let bestCols = 1;
  let bestWidth = 0;
  for (let cols = 1; cols <= count; cols++) {
    const rows = Math.ceil(count / cols);
    const byWidth = (width - gap * (cols - 1)) / cols;
    const byHeight = ((height - gap * (rows - 1)) / rows) * aspect;
    const tileWidth = Math.min(byWidth, byHeight);
    if (tileWidth > bestWidth) {
      bestWidth = tileWidth;
      bestCols = cols;
    }
  }
  const tileWidth = Math.max(0, Math.floor(bestWidth));
  return { cols: bestCols, tileWidth, tileHeight: Math.floor(tileWidth / aspect) };
}
