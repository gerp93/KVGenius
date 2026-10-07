import { useEffect, useState } from 'react';
import { ModelFileCheck, ModelImportOutcome } from '../../shared/modelCheck';
import { ModelSlot } from '../../shared/modelFamilies';
import ImageDropZone from './ImageDropZone';

interface Props {
  family: string;
  slot: ModelSlot;
  /** Whether ComfyUI's models folder is known - files are copied into it. */
  canImport: boolean;
  /** The file is in ComfyUI's folder now (under this name): select it and look again at what ComfyUI has. */
  onImported: (fileName: string) => void;
}

function formatBytes(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

type Phase =
  | { kind: 'idle' }
  | { kind: 'checking'; path: string }
  | { kind: 'checked'; path: string; check: ModelFileCheck }
  | { kind: 'importing'; path: string; copied: number; total: number }
  | { kind: 'exists'; path: string; check: ModelFileCheck; message: string }
  | { kind: 'done'; fileName: string; originalKept: boolean }
  | { kind: 'failed'; message: string };

/**
 * "Use a file I already have": pick (or drop) a model file, see whether it looks right, and have it copied into
 * ComfyUI's folder for this slot. The file is looked over before anything is copied, so a wrong or cut-off one
 * never lands in ComfyUI's folder.
 */
export default function ModelFileImport({ family, slot, canImport, onImported }: Props) {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [move, setMove] = useState(false);

  useEffect(() => window.kvgenius.onModelImportProgress((p) => setPhase((cur) => (cur.kind === 'importing' ? { ...cur, copied: p.copied, total: p.total } : cur))), []);

  async function checkFile(path: string) {
    setPhase({ kind: 'checking', path });
    try {
      setPhase({ kind: 'checked', path, check: await window.kvgenius.checkModelFile(path, family, slot.key) });
    } catch (err) {
      setPhase({ kind: 'failed', message: err instanceof Error ? err.message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '') : String(err) });
    }
  }

  async function choose() {
    const picked = await window.kvgenius.chooseModelFile();
    if (picked) await checkFile(picked.path);
  }

  async function runImport(path: string, check: ModelFileCheck, overwrite: boolean) {
    setPhase({ kind: 'importing', path, copied: 0, total: check.sizeBytes });
    const outcome: ModelImportOutcome = await window.kvgenius.importModelFile(path, family, slot.key, { move, overwrite });
    if (outcome.ok) {
      setPhase({ kind: 'done', fileName: outcome.fileName, originalKept: outcome.originalKept });
      onImported(outcome.fileName);
    } else if (outcome.code === 'exists') {
      setPhase({ kind: 'exists', path, check, message: outcome.message });
    } else if (outcome.code === 'cancelled') {
      setPhase({ kind: 'idle' });
    } else {
      setPhase({ kind: 'failed', message: outcome.message });
    }
  }

  return (
    <ImageDropZone
      kind="model"
      className="model-import"
      onPaths={(paths) => void checkFile(paths[0])}
      onReject={(message) => setPhase({ kind: 'failed', message })}
    >
      <div className="model-import__bar">
        <button type="button" onClick={() => void choose()} disabled={phase.kind === 'importing' || phase.kind === 'checking'}>
          Use a file from this computer...
        </button>
        <span className="model-import__hint">or drop it here</span>
      </div>

      {phase.kind === 'checking' && <p className="model-import__note">Looking the file over...</p>}

      {(phase.kind === 'checked' || phase.kind === 'exists') && (
        <div className={`model-import__result model-import__result--${phase.check.severity}`}>
          <strong>
            {phase.check.fileName} ({formatBytes(phase.check.sizeBytes)})
          </strong>
          <ul>
            {phase.check.messages.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
          {phase.kind === 'exists' && <p>{phase.message} Replace it?</p>}
          {!canImport && phase.check.severity !== 'block' && <p className="model-import__warn">Set ComfyUI's models folder (above) before importing.</p>}
          <div className="button-row">
            {phase.check.severity !== 'block' && (
              <button
                type="button"
                className="primary"
                disabled={!canImport}
                onClick={() => void runImport(phase.path, phase.check, phase.kind === 'exists')}
              >
                {phase.kind === 'exists' ? 'Replace it' : phase.check.severity === 'warn' ? 'Import anyway' : 'Import'}
              </button>
            )}
            <label className="model-import__move">
              <input type="checkbox" checked={move} onChange={(e) => setMove(e.target.checked)} /> Move it (remove the original afterwards)
            </label>
            <button type="button" onClick={() => setPhase({ kind: 'idle' })}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {phase.kind === 'importing' && (
        <div className="model-import__result">
          <progress value={phase.copied} max={Math.max(1, phase.total)} />
          <span>
            {formatBytes(phase.copied)} of {formatBytes(phase.total)}
          </span>
          <button type="button" onClick={() => void window.kvgenius.cancelModelImport()}>
            Cancel
          </button>
        </div>
      )}

      {phase.kind === 'done' && (
        <p className="model-import__note">
          Copied {phase.fileName} into {slot.folder}.{phase.originalKept ? ' The original could not be removed.' : ''} If ComfyUI does not list it yet,
          restart it (or press R in its window) and use Check Again.
        </p>
      )}
      {phase.kind === 'failed' && <p className="model-import__warn">{phase.message}</p>}
    </ImageDropZone>
  );
}
