/**
 * Which batch of results the Generate page shows for each working tab. Every tab keeps its own
 * picture: a generation finishing for another tab never replaces it. Jobs with no tab (an upscale
 * started from the Library) belong to no tab's viewer.
 */

/** The part of a queue job these rules need. */
export interface SlotJob {
  batchId: number;
  /** The working tab that queued it; none for jobs from elsewhere (e.g. Library upscales). */
  slotId?: string;
  status: 'queued' | 'running' | 'done' | 'failed';
}

/** The batch each tab has been showing, by tab id. */
export type SlotViews = Readonly<Record<string, number>>;

const isActive = (job: SlotJob) => job.status === 'queued' || job.status === 'running';

/** The batch this tab should show: the one it was last pointed at, else (that batch is gone, e.g.
 * cancelled away) the newest of its own, else nothing - a tab with no results shows no picture. */
export function batchShownFor(jobs: readonly SlotJob[], views: SlotViews, slotId: string): number | null {
  const own = jobs.filter((j) => j.slotId === slotId);
  const viewed = views[slotId];
  if (viewed !== undefined && own.some((j) => j.batchId === viewed)) return viewed;
  return own.length > 0 ? own[own.length - 1].batchId : null;
}

/** Whether a tab's viewer should move on to a newly queued or newly started batch of its own: yes
 * unless it is showing a batch that is still being worked on (the user may be paging through it). */
export function shouldFollow(jobs: readonly SlotJob[], views: SlotViews, slotId: string): boolean {
  const shown = batchShownFor(jobs, views, slotId);
  return shown === null || !jobs.some((j) => j.slotId === slotId && j.batchId === shown && isActive(j));
}
