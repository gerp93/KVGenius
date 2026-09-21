import { TimingInfo } from '../../shared/types';
import { formatDifference, formatDuration } from './format';

/** Model loading shorter than this is just setup noise, not worth mentioning. */
const NOTABLE_LOAD_MS = 2000;

/** One line: what was estimated, what it took, and how far off the estimate was. */
export function timingSentence(timing: TimingInfo): string {
  const loading =
    timing.loadMs !== null && timing.loadMs >= NOTABLE_LOAD_MS
      ? ` (including ${formatDuration(timing.loadMs)} loading models)`
      : '';
  if (timing.estimateMs === null) {
    return `Took ${formatDuration(timing.actualMs)}${loading} - no estimate yet`;
  }
  return `Estimated ${formatDuration(timing.estimateMs)} · took ${formatDuration(timing.actualMs)}${loading} · ${formatDifference(timing.estimateMs, timing.actualMs)}`;
}
