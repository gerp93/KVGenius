import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ModelsDirInfo } from '../../shared/types';
import ModelProfiles from '../components/ModelProfiles';
import { useModelStatus } from '../hooks/useModelStatus';
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

        <ModelProfiles report={report} onChanged={onModelsChanged} canImport={modelsDir !== null} onFilesChanged={refresh} modelsDir={modelsDir} />
    </div>
  );
}
