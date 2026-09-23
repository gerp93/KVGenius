import { ReactNode, useEffect, useMemo, useState } from 'react';
import { TimingStatRow } from '../../shared/types';
import {
  AccuracySummary,
  Metric,
  groupBySettings,
  pairFor,
  diffPct,
  recentVsEarlier,
  settingsLabel,
  summarize,
  trend,
} from '../../shared/timingAnalysis';
import { formatDuration } from '../utils/format';

const pct = (n: number) => `${Math.round(n)}%`;
const signedPct = (n: number) => `${n > 0 ? '+' : n < 0 ? '-' : ''}${Math.abs(Math.round(n))}%`;

/** A signed percentage, with negative values in the app's "negative" color (the same red used
 * for errors and "slower than estimated" elsewhere on this page) so a run of - and + numbers in
 * a table is easy to scan at a glance. */
function Signed({ value }: { value: number }) {
  return <span className={value < 0 ? 'timing-negative' : undefined}>{signedPct(value)}</span>;
}

/** "usually 8% slower than estimated" - the direction of the average miss in words. */
function biasWords(biasPct: number): string {
  if (Math.abs(biasPct) < 2) return 'no consistent lean';
  return `usually ${Math.abs(Math.round(biasPct))}% ${biasPct > 0 ? 'slower' : 'faster'} than estimated`;
}

function StatCard({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="timing-card">
      <div className="timing-card__label">{label}</div>
      <div className="timing-card__value">{value}</div>
      {hint && <div className="timing-card__hint">{hint}</div>}
    </div>
  );
}

function TrendChart({ rows, metric }: { rows: TimingStatRow[]; metric: Metric }) {
  const points = trend(rows, metric, 60);
  if (points.length < 2) return <p className="timing-muted">Needs at least two runs with an estimate.</p>;
  const barWidth = 10;
  const height = 160;
  const half = height / 2;
  const clamp = 100; // bars are clipped at +/-100% so one wild run does not flatten the rest
  return (
    <svg
      className="timing-chart"
      viewBox={`0 0 ${points.length * barWidth} ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="How far each recent run was from its estimate"
    >
      <line x1={0} x2={points.length * barWidth} y1={half} y2={half} className="timing-chart__zero" />
      {points.map((p, i) => {
        const value = Math.max(-clamp, Math.min(clamp, p.diffPct));
        const h = (Math.abs(value) / clamp) * (half - 4);
        return (
          <rect
            key={p.id}
            x={i * barWidth + 1}
            width={barWidth - 2}
            y={value >= 0 ? half - h : half}
            height={Math.max(1, h)}
            className={value >= 0 ? 'timing-chart__slower' : 'timing-chart__faster'}
          >
            <title>{`${new Date(p.createdAt).toLocaleString()}: ${signedPct(p.diffPct)}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export default function Timing() {
  const [rows, setRows] = useState<TimingStatRow[] | null>(null);
  const [metric, setMetric] = useState<Metric>('total');
  const [error, setError] = useState<string | null>(null);

  function load() {
    window.kvgenius
      .getTimingStats()
      .then(setRows)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }
  useEffect(load, []);

  const summary = useMemo(() => (rows ? summarize(rows, metric) : null), [rows, metric]);
  const improving = useMemo(() => (rows ? recentVsEarlier(rows, metric) : null), [rows, metric]);
  const groups = useMemo(() => (rows ? groupBySettings(rows, metric) : []), [rows, metric]);
  const withEstimate = rows ? rows.filter((r) => pairFor(r, metric) !== null) : [];
  const warmSplit = useMemo(() => {
    if (!rows) return null;
    const side = (warm: boolean) => {
      const list = rows.filter((r) => r.warm === warm);
      const loads = list.map((r) => r.loadMs).filter((l): l is number => l !== null);
      return {
        runs: list.length,
        avgLoad: loads.length ? loads.reduce((a, b) => a + b, 0) / loads.length : null,
        summary: summarize(list, metric),
      };
    };
    return { warm: side(true), cold: side(false) };
  }, [rows, metric]);

  async function handleClear() {
    if (!window.confirm('Delete all timing data? Future estimates start over and learn from scratch.')) return;
    try {
      await window.kvgenius.clearTimingStats();
      load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="page timing-page">
      <div className="timing-header">
        <div>
          <h2 style={{ margin: 0 }}>Estimate accuracy</h2>
          <p className="timing-muted" style={{ margin: '4px 0 0' }}>
            How close the time estimates were to how long generations really took. This is built only from timings and
            the size / steps / CFG / video length - nothing about prompts or images is stored here, and it is kept even
            if you delete the images.
          </p>
        </div>
        <div className="timing-header__actions">
          <div className="timing-toggle" role="group" aria-label="Which time to compare">
            <button type="button" className={metric === 'total' ? 'primary' : undefined} onClick={() => setMetric('total')}>
              Total time
            </button>
            <button
              type="button"
              className={metric === 'generate' ? 'primary' : undefined}
              onClick={() => setMetric('generate')}
              title="Leaves out the time spent loading models into memory"
            >
              Excluding model loading
            </button>
          </div>
          <button type="button" onClick={handleClear} disabled={!rows || rows.length === 0}>
            Delete timing data
          </button>
        </div>
      </div>

      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

      {rows && rows.length === 0 && (
        <p className="timing-muted">
          No timings recorded yet. Generate something - each finished run is recorded here, and estimates start
          appearing after the first one.
        </p>
      )}

      {rows && rows.length > 0 && !summary && (
        <p className="timing-muted">
          {rows.length} run{rows.length === 1 ? '' : 's'} recorded, but none had an estimate to compare against yet (the
          first runs of each kind have none).
        </p>
      )}

      {summary && (
        <>
          <div className="timing-cards">
            <StatCard label="Runs compared" value={String(summary.runs)} hint={`of ${rows?.length ?? 0} recorded`} />
            <StatCard label="Typical miss" value={pct(summary.medianAbsPct)} hint="median, either direction" />
            <StatCard label="Average miss" value={pct(summary.meanAbsPct)} hint="pulled up by outliers" />
            <StatCard label="Lean" value={<Signed value={summary.biasPct} />} hint={biasWords(summary.biasPct)} />
            <StatCard label="Within 10%" value={pct(summary.within10Pct * 100)} hint="of runs" />
            <StatCard label="Within 25%" value={pct(summary.within25Pct * 100)} hint="of runs" />
            <StatCard
              label="Average estimate"
              value={formatDuration(summary.avgEstimateMs)}
              hint={`vs average actual ${formatDuration(summary.avgActualMs)}`}
            />
            {improving && (
              <StatCard
                label="Getting better?"
                value={improving.recent < improving.earlier ? 'Yes' : 'Not yet'}
                hint={`last 10 runs miss by ${pct(improving.recent)}, the 10 before by ${pct(improving.earlier)}`}
              />
            )}
          </div>

          <h3 className="timing-h3">Each recent run, against its estimate</h3>
          <TrendChart rows={rows ?? []} metric={metric} />
          <p className="timing-muted timing-legend">
            <span className="timing-swatch timing-swatch--slower" /> took longer than estimated
            <span className="timing-swatch timing-swatch--faster" /> faster than estimated. Oldest on the left, newest on
            the right; bars stop at 100%.
          </p>

          <h3 className="timing-h3">By settings</h3>
          <table className="timing-table">
            <thead>
              <tr>
                <th>Settings</th>
                <th>Runs</th>
                <th>Avg estimate</th>
                <th>Avg actual</th>
                <th>Lean</th>
                <th>Typical miss</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <tr key={g.label}>
                  <td>{g.label}</td>
                  <td>{g.runs}</td>
                  <td>{g.summary ? formatDuration(g.summary.avgEstimateMs) : '-'}</td>
                  <td>{g.summary ? formatDuration(g.summary.avgActualMs) : '-'}</td>
                  <td>{g.summary ? <Signed value={g.summary.biasPct} /> : '-'}</td>
                  <td>{g.summary ? pct(g.summary.medianAbsPct) : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {warmSplit && (
            <>
              <h3 className="timing-h3">Models already loaded vs. freshly loaded</h3>
              <table className="timing-table">
                <thead>
                  <tr>
                    <th>Situation</th>
                    <th>Runs</th>
                    <th>Avg model load time</th>
                    <th>Typical miss</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ['Same model as the previous run', warmSplit.warm],
                      ['First run / switched model', warmSplit.cold],
                    ] as [string, { runs: number; avgLoad: number | null; summary: AccuracySummary | null }][]
                  ).map(([label, s]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      <td>{s.runs}</td>
                      <td>{s.avgLoad === null ? '-' : formatDuration(s.avgLoad)}</td>
                      <td>{s.summary ? pct(s.summary.medianAbsPct) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}

      {rows && rows.length > 0 && (
        <>
          <h3 className="timing-h3">Recent runs</h3>
          <table className="timing-table">
            <thead>
              <tr>
                <th>When</th>
                <th>Settings</th>
                <th>Estimate</th>
                <th>Actual</th>
                <th>Difference</th>
                <th>Model loading</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 25).map((r) => {
                const pair = pairFor(r, metric);
                return (
                  <tr key={r.id}>
                    <td>{new Date(r.createdAt).toLocaleString()}</td>
                    <td>{settingsLabel(r)}</td>
                    <td>{pair ? formatDuration(pair.estimate) : 'none yet'}</td>
                    <td>{formatDuration(metric === 'total' ? r.actualMs : (r.generateMs ?? r.actualMs))}</td>
                    <td>{pair ? <Signed value={diffPct(pair.estimate, pair.actual)} /> : '-'}</td>
                    <td>{r.loadMs === null ? '-' : formatDuration(r.loadMs)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {withEstimate.length < rows.length && (
            <p className="timing-muted">
              {rows.length - withEstimate.length} run{rows.length - withEstimate.length === 1 ? ' has' : 's have'} no
              estimate to compare (made before there was history to go on).
            </p>
          )}
        </>
      )}
    </div>
  );
}
