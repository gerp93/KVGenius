import { useRef } from 'react';
import { Job, ProgressInfo } from '../hooks/useGenerationQueue';
import { formatDuration, formatElapsed } from '../utils/format';
import { describeProgress } from '../utils/progressModel';

interface Props {
  job: Job;
  progressInfo: ProgressInfo | null;
  now: number;
  /** Smaller text for the queue panel. */
  compact?: boolean;
}

/** What the running generation is doing: stage text, a bar, and time elapsed / left. */
export default function RunProgress({ job, progressInfo, now, compact }: Props) {
  const startedAt = job.startedAt ?? now;
  const elapsedMs = Math.max(0, now - startedAt);
  const display = describeProgress({
    progress: progressInfo?.progress ?? null,
    estimate: job.estimate ?? null,
    elapsedMs,
    lastStepAtMs: progressInfo?.lastStepAt != null ? progressInfo.lastStepAt - startedAt : null,
  });

  // The bar only ever moves forward, so a late correction never makes it jump backwards.
  const highest = useRef({ jobId: job.id, value: 0 });
  if (highest.current.jobId !== job.id) highest.current = { jobId: job.id, value: 0 };
  if (display.determinate) highest.current.value = Math.max(highest.current.value, display.fraction);
  const fraction = display.determinate ? highest.current.value : 0;

  let timeText = '';
  if (display.overrunning) timeText = 'taking longer than usual';
  else if (display.remainingMs !== null) timeText = `about ${formatDuration(display.remainingMs)} left`;

  return (
    <div className={`run-progress${compact ? ' run-progress--compact' : ''}`}>
      <div className="run-progress__label">{display.label}</div>
      <div className={`progress-bar${display.determinate ? '' : ' progress-bar--indeterminate'}`}>
        {display.determinate && <div className="progress-bar__fill" style={{ width: `${fraction * 100}%` }} />}
      </div>
      <div className="run-progress__meta">
        {formatElapsed(Math.floor(elapsedMs / 1000))} elapsed{timeText ? ` · ${timeText}` : ''}
      </div>
    </div>
  );
}
