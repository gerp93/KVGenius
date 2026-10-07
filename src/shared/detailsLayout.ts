/** Width of the details column beside the picture in the two-column layout. */
export const SPLIT_INFO_WIDTH = 360;
/** Padding around the picture, in either layout. */
const IMAGE_PADDING = 12;
/** Narrowest picture column worth having. */
const MIN_IMAGE_COLUMN = 260;
/** Roughly the height of everything that sits above and below the picture in the stacked layout
 * (header, actions, upscale, a short prompt, the metadata) - what the picture has to leave room for. */
const STACKED_OVERHEAD = 480;
/** The stacked picture never gets shorter than this (see `.library-panel__media`). */
const STACKED_MIN_IMAGE_HEIGHT = 380;
/** The split layout has to show the picture at least this much larger to be worth switching to. */
const SPLIT_GAIN = 1.1;

export interface PanelShape {
  /** The panel's size as it is (its width follows the window; it runs the full height). */
  panelWidth: number;
  panelHeight: number;
  /** The picture's width / height. */
  aspect: number;
}

/** The area a picture of this shape covers when fitted inside a box. */
function fittedArea(boxWidth: number, boxHeight: number, aspect: number): number {
  if (boxWidth <= 0 || boxHeight <= 0) return 0;
  const width = Math.min(boxWidth, boxHeight * aspect);
  return width * (width / aspect);
}

/**
 * Whether the details panel should show its two-column layout - the picture in a column of its own at
 * the panel's full height, the details in a scrolling column beside it - instead of stacking the
 * picture above the details. The panel's width is not changed by this; it is chosen only when the panel
 * is big enough that the picture comes out clearly larger that way, which depends on the picture's
 * shape: a tall one gains a lot from the full height, a wide one often does not.
 */
export function shouldSplitDetails({ panelWidth, panelHeight, aspect }: PanelShape): boolean {
  if (!(aspect > 0) || panelHeight <= 0) return false;
  const pad = 2 * IMAGE_PADDING;
  const columnWidth = panelWidth - SPLIT_INFO_WIDTH - pad;
  if (columnWidth < MIN_IMAGE_COLUMN) return false;
  const split = fittedArea(columnWidth, panelHeight - pad, aspect);
  const stacked = fittedArea(panelWidth - pad, Math.max(STACKED_MIN_IMAGE_HEIGHT, panelHeight - STACKED_OVERHEAD), aspect);
  return split > stacked * SPLIT_GAIN;
}
