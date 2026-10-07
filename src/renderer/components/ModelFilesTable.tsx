import { ManifestFeature } from '../../shared/modelManifest';
import { fileState, ModelStatusReport, summarize } from '../../shared/modelStatus';
import './Models.css';

interface Props {
  feature: ManifestFeature;
  /** null while the first check is running. */
  report: ModelStatusReport | null;
  /** Where the models folder is, if known, so a missing file can be given its exact destination. */
  modelsDir: string | null;
  /** Hide the status column (a guide that only lists what to get). */
  showStatus?: boolean;
}

const STATE_LABEL = {
  present: '✓ Installed',
  missing: '✗ Missing',
  'in-subfolder': '⚠ In a subfolder',
  unknown: '-',
} as const;

/** The files a feature needs: name, role, the folder it goes in, and - when known - whether it is there. */
export default function ModelFilesTable({ feature, report, modelsDir, showStatus = true }: Props) {
  if (feature.files.length === 0) return null;
  const known = report !== null && report.source !== 'none';
  return (
    <table className="models-table">
      <thead>
        <tr>
          <th>File</th>
          <th>Goes in</th>
          {showStatus && <th>Status</th>}
        </tr>
      </thead>
      <tbody>
        {feature.files.map((file) => {
          const state = report ? fileState(file, report) : null;
          return (
            <tr key={file.file}>
              <td>
                <code>{file.file}</code>
                <div className="models-table__role">{file.role}</div>
              </td>
              <td>
                <code>{file.folder}</code>
                {modelsDir && state?.state === 'missing' && <div className="models-table__role">{modelsDir}</div>}
              </td>
              {showStatus && (
                <td className={`models-table__state models-table__state--${state?.state ?? 'unknown'}`}>
                  {known && state ? STATE_LABEL[state.state] : STATE_LABEL.unknown}
                  {state?.state === 'in-subfolder' && (
                    <div className="models-table__role">
                      Found as <code>{state.foundAs}</code>. Move it straight into <code>{file.folder}</code> - the workflow asks for the
                      plain name.
                    </div>
                  )}
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** The files of a feature whose model the user picks themselves (upscaling): the same table as above, listing
 * what ComfyUI has in `folder` - or a note saying where to put one when there is none. */
export function ChosenModelsTable({ folder, role, names, emptyText }: { folder: string; role: string; names: string[]; emptyText: string }) {
  if (names.length === 0) return <p className="settings-hint">{emptyText}</p>;
  return (
    <table className="models-table">
      <thead>
        <tr>
          <th>File</th>
          <th>Goes in</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {names.map((name) => (
          <tr key={name}>
            <td>
              <code>{name}</code>
              <div className="models-table__role">{role}</div>
            </td>
            <td>
              <code>{folder}</code>
            </td>
            <td className="models-table__state models-table__state--present">{STATE_LABEL.present}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** "3 of 3 installed" style line for a feature; null when nothing is known. */
export function summaryText(feature: ManifestFeature, report: ModelStatusReport | null): string | null {
  if (!report || report.source === 'none' || feature.files.length === 0) return null;
  const s = summarize(feature.files, report);
  if (s.present === s.total) return `All ${s.total} installed`;
  const parts = [`${s.present} of ${s.total} installed`];
  if (s.inSubfolder > 0) parts.push(`${s.inSubfolder} in a subfolder`);
  return parts.join(', ');
}
