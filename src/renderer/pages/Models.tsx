import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { MODEL_MANIFEST } from '../../shared/modelManifest';
import { ModelsDirInfo } from '../../shared/types';
import { ExternalLink } from '../components/Stepper';
import ModelFilesTable, { summaryText } from '../components/ModelFilesTable';
import ModelProfiles from '../components/ModelProfiles';
import { useModelStatus } from '../hooks/useModelStatus';
import SettingsSection from './settings/SettingsSection';
import '../components/Models.css';

const SOURCE_TEXT = {
  comfyui: 'Read from ComfyUI, so this is exactly what it can load.',
  folder: "ComfyUI is not reachable, so this was read from the models folder. Start ComfyUI to confirm.",
  none: "Can't tell yet: ComfyUI is not reachable and its models folder is not set.",
} as const;

interface Props {
  /** Tells the app a saved model was added, edited or deleted, so Generate's dropdown reloads. */
  onModelsChanged: () => void;
}

/** What model files KVGenius needs, which of them ComfyUI has, and where the missing ones go. */
export default function Models({ onModelsChanged }: Props) {
  const { report, refresh } = useModelStatus();
  const [dir, setDir] = useState<ModelsDirInfo | null>(null);

  useEffect(() => {
    void window.kvgenius.getModelsDirInfo().then(setDir);
  }, []);

  // The folder can change what the status says (no ComfyUI to ask), so recheck after any change.
  async function changeDir(action: () => Promise<ModelsDirInfo | null>) {
    const next = await action();
    if (next) setDir(next);
    refresh();
  }

  const modelsDir = dir?.valid ? dir.effective : null;

  return (
    <div className="page">
      <div className="models-page">
        <h2 style={{ marginTop: 0 }}>Models</h2>
        <p className="models-page__status">
          {report ? SOURCE_TEXT[report.source] : 'Checking...'} <Link to="/setup">Setup guide</Link>
        </p>

        <SettingsSection
          title="ComfyUI's models folder"
          description="Where the files below go. Found automatically for a portable install; ComfyUI Desktop keeps it in the base folder you chose when installing, so you may need to pick it."
        >
          <p style={{ fontSize: 12, wordBreak: 'break-all', marginTop: 0 }}>
            {dir === null
              ? '...'
              : dir.effective
                ? `${dir.configured ? 'Chosen' : 'Found automatically'}: ${dir.effective}${dir.valid ? '' : ' (this does not look like a ComfyUI models folder)'}`
                : 'Not set.'}
          </p>
          <div className="button-row">
            <button type="button" onClick={() => void changeDir(() => window.kvgenius.chooseModelsDir())}>
              Choose Folder...
            </button>
            {dir?.configured && (
              <button type="button" onClick={() => void changeDir(() => window.kvgenius.clearModelsDir())}>
                Clear
              </button>
            )}
            <button type="button" onClick={refresh}>
              Check Again
            </button>
          </div>
        </SettingsSection>

        <ModelProfiles report={report} onChanged={onModelsChanged} canImport={modelsDir !== null} onFilesChanged={refresh} />

        {MODEL_MANIFEST.map((feature) => {
          const summary = summaryText(feature, report);
          const upscaleModels = feature.id === 'upscale' && report && report.source !== 'none' ? report.installed.upscale_models : null;
          return (
            <SettingsSection key={feature.id} title={feature.title} description={feature.summary}>
              {summary && <div className="models-feature__summary">{summary}</div>}
              <ModelFilesTable feature={feature} report={report} modelsDir={modelsDir} />
              {upscaleModels && (
                <p style={{ fontSize: 13 }}>
                  {upscaleModels.length > 0 ? `Installed: ${upscaleModels.join(', ')}` : 'No upscale models installed yet. Put one in upscale_models.'}
                </p>
              )}
              {feature.note && <p className="settings-hint">{feature.note}</p>}
              <ExternalLink href={feature.source.url}>{feature.source.label} ↗</ExternalLink>
            </SettingsSection>
          );
        })}
      </div>
    </div>
  );
}
