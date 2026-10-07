import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { MODEL_MANIFEST } from '../../shared/modelManifest';
import { ModelsDirInfo } from '../../shared/types';
import { ExternalLink } from '../components/Stepper';
import ModelFilesTable, { ChosenModelsTable, summaryText } from '../components/ModelFilesTable';
import ModelDownload from '../components/ModelDownload';
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

/** What model files KVGenius needs, which of them ComfyUI has, and where the missing ones go. Shown as Settings' Models tab. */
export default function Models({ onModelsChanged }: Props) {
  const { report, refresh } = useModelStatus();
  const [dir, setDir] = useState<ModelsDirInfo | null>(null);

  useEffect(() => {
    void window.kvgenius.getModelsDirInfo().then(setDir);
  }, []);

  const modelsDir = dir?.valid ? dir.effective : null;

  return (
    <div className="models-page">
        <p className="models-page__status">
          {report ? SOURCE_TEXT[report.source] : 'Checking...'} <Link to="/setup">Setup guide</Link>
        </p>

        {dir && !dir.valid && (
          <p className="settings-message settings-message--error" style={{ marginBottom: 16 }}>
            {dir.effective ? "The models folder doesn't look right" : "The models folder isn't set"}, so files can't be downloaded or added for you.{' '}
            <Link to="/settings?tab=comfyui">Set it in Settings &gt; ComfyUI</Link>
          </p>
        )}

        <ModelProfiles report={report} onChanged={onModelsChanged} canImport={modelsDir !== null} onFilesChanged={refresh} />

        {MODEL_MANIFEST.map((feature) => {
          const summary = summaryText(feature, report);
          const upscaleModels = feature.id === 'upscale' && report && report.source !== 'none' ? report.installed.upscale_models : null;
          return (
            <SettingsSection key={feature.id} title={feature.title} description={feature.summary}>
              {summary && <div className="models-feature__summary">{summary}</div>}
              <ModelFilesTable feature={feature} report={report} modelsDir={modelsDir} />
              <ModelDownload feature={feature} report={report} modelsDir={modelsDir} />
              {upscaleModels && (
                <ChosenModelsTable
                  folder="upscale_models"
                  role="Upscale model"
                  names={upscaleModels}
                  emptyText="No upscale models installed yet. Put one in upscale_models."
                />
              )}
              {feature.note && <p className="settings-hint">{feature.note}</p>}
              <ExternalLink href={feature.source.url}>{feature.source.label} ↗</ExternalLink>
            </SettingsSection>
          );
        })}
    </div>
  );
}
