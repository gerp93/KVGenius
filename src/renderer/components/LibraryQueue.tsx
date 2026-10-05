import { GenerationRecord } from '../../shared/types';
import { isUpscaleFamily } from '../../shared/upscale';
import { GenerationQueue } from '../hooks/useGenerationQueue';
import QueuePanel from './QueuePanel';

interface Props {
  queue: GenerationQueue;
  collapsed: boolean;
  onToggle: () => void;
  onToggleFavorite: (record: GenerationRecord) => void;
  onRerack: (record: GenerationRecord) => void;
}

/** The queue beside a Library page: pops out while anything is running or waiting, and stays for
 * finished or failed upscales. Nothing at all otherwise. */
export default function LibraryQueue({ queue, collapsed, onToggle, onToggleFavorite, onRerack }: Props) {
  const show = queue.jobs.some(
    (j) =>
      j.status === 'queued' ||
      j.status === 'running' ||
      (isUpscaleFamily(j.family) && (j.status === 'done' || (j.status === 'failed' && !j.dismissed)))
  );
  if (!show) return null;
  return (
    <div className={`library-queue${collapsed ? ' library-queue--collapsed' : ''}`}>
      <QueuePanel
        jobs={queue.jobs}
        now={queue.now}
        progressInfo={queue.progressInfo}
        collapsed={collapsed}
        onToggle={onToggle}
        onCancelJob={queue.cancelJob}
        onClearQueued={queue.clearQueued}
        onDismissFailed={queue.dismissFailed}
        onToggleFavorite={onToggleFavorite}
        onRerack={onRerack}
      />
    </div>
  );
}
