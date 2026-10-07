import { useState } from 'react';
import { DownloadPlanInfo, DownloadStartResult } from '../../shared/modelDownloads';
import { ManifestFeature } from '../../shared/modelManifest';
import { ModelStatusReport, summarize } from '../../shared/modelStatus';
import CopyButton from './CopyButton';

interface Props {
  feature: ManifestFeature;
  report: ModelStatusReport | null;
  /** ComfyUI's models folder, if known - the files are saved into it. */
  modelsDir: string | null;
}

type Phase =
  | { kind: 'idle' }
  | { kind: 'planning' }
  | { kind: 'confirm'; plan: DownloadPlanInfo }
  | { kind: 'starting' }
  | { kind: 'launched'; scriptPath: string }
  | { kind: 'copy'; result: Extract<DownloadStartResult, { status: 'copy' }> }
  | { kind: 'problem'; message: string };

function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/**
 * "Download missing files" for a feature: shows exactly what would be fetched and from where, and only after a
 * confirmation opens a terminal window that does it (progress shown there; running it again resumes).
 */
export default function ModelDownload({ feature, report, modelsDir }: Props) {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  if (feature.files.length === 0 || !feature.files.some((f) => f.url)) return null;
  const missing = report && report.source !== 'none' ? summarize(feature.files, report).missing : 0;
  if (missing === 0 && phase.kind === 'idle') return null;

  async function plan() {
    setPhase({ kind: 'planning' });
    try {
      const result = await window.kvgenius.planModelDownloads([feature.id]);
      setPhase(result.problem ? { kind: 'problem', message: result.problem } : { kind: 'confirm', plan: result });
    } catch (err) {
      setPhase({ kind: 'problem', message: cleanError(err) });
    }
  }

  async function start() {
    setPhase({ kind: 'starting' });
    try {
      const result = await window.kvgenius.startModelDownloads([feature.id]);
      if (result.status === 'launched') setPhase({ kind: 'launched', scriptPath: result.scriptPath });
      else if (result.status === 'copy') setPhase({ kind: 'copy', result });
      else setPhase({ kind: 'problem', message: result.message });
    } catch (err) {
      setPhase({ kind: 'problem', message: cleanError(err) });
    }
  }

  return (
    <div className="model-download">
      {phase.kind === 'idle' && (
        <div className="button-row">
          <button type="button" className="primary" onClick={() => void plan()} disabled={!modelsDir} title={modelsDir ? undefined : "Set ComfyUI's models folder first"}>
            Download {missing} missing file{missing === 1 ? '' : 's'}...
          </button>
          {!modelsDir && <span className="settings-hint" style={{ margin: 0 }}>Set ComfyUI's models folder above first - the files are saved into it.</span>}
        </div>
      )}
      {phase.kind === 'planning' && <p className="settings-hint">Working out what is missing...</p>}

      {phase.kind === 'confirm' && (
        <div className="model-download__confirm">
          <strong>
            Download {phase.plan.items.length} file{phase.plan.items.length === 1 ? '' : 's'}?
          </strong>
          <table className="models-table">
            <tbody>
              {phase.plan.items.map((item) => (
                <tr key={item.destPath}>
                  <td>
                    <code>{item.fileName}</code>
                    <div className="models-table__role">{item.label}</div>
                  </td>
                  <td>
                    from {hostOf(item.url)}
                    <div className="models-table__role">into {item.destPath}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {phase.plan.inSubfolder.length > 0 && (
            <p className="settings-hint">Not included, already there in a subfolder (move up a level): {phase.plan.inSubfolder.join(', ')}.</p>
          )}
          <p className="settings-hint">
            These are large files. A terminal window opens and shows the progress; if it is interrupted, run it again and it carries on where it
            stopped. Nothing here needs an account.
          </p>
          <div className="button-row">
            <button type="button" className="primary" onClick={() => void start()}>
              Start download
            </button>
            <button type="button" onClick={() => setPhase({ kind: 'idle' })}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {phase.kind === 'starting' && <p className="settings-hint">Opening a terminal...</p>}

      {phase.kind === 'launched' && (
        <p className="settings-hint">
          A terminal window opened and is downloading. When it finishes, ComfyUI may need a restart (or press R in its window) before it lists the
          files; this page then updates by itself. To resume an interrupted download, run <code>{phase.scriptPath}</code> again.{' '}
          <button type="button" onClick={() => setPhase({ kind: 'idle' })}>
            OK
          </button>
        </p>
      )}

      {phase.kind === 'copy' && (
        <div className="model-download__confirm">
          <p>
            {phase.result.reason} Run this script yourself in a terminal (<code>{phase.result.scriptPath}</code>), or copy it:
          </p>
          <pre className="model-download__script">{phase.result.script}</pre>
          <div className="button-row">
            <CopyButton text={phase.result.script} label="Copy script" />
            <button type="button" onClick={() => setPhase({ kind: 'idle' })}>
              Done
            </button>
          </div>
        </div>
      )}

      {phase.kind === 'problem' && (
        <p className="model-import__warn">
          {phase.message}{' '}
          <button type="button" onClick={() => setPhase({ kind: 'idle' })}>
            OK
          </button>
        </p>
      )}
    </div>
  );
}
