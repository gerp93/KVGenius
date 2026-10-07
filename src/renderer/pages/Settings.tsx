import { useSearchParams } from 'react-router-dom';
import GeneralTab from './settings/GeneralTab';
import ComfyUITab from './settings/ComfyUITab';
import LibraryDataTab from './settings/LibraryDataTab';
import IntegrationsTab from './settings/IntegrationsTab';
import Models from './Models';

const TABS = [
  { id: 'general', label: 'General' },
  { id: 'comfyui', label: 'ComfyUI' },
  { id: 'models', label: 'Models' },
  { id: 'library', label: 'Library & Data' },
  { id: 'integrations', label: 'Integrations' },
] as const;

type TabId = (typeof TABS)[number]['id'];

interface Props {
  theme: string | null;
  onThemeChange: (theme: string) => void;
  /** Tells the app a saved model was added, edited or deleted, so Generate's dropdown reloads. */
  onModelsChanged: () => void;
}

/** Settings, grouped into tabs. The active tab lives in the URL (`/settings?tab=comfyui`) so other
 * parts of the app can link straight to the relevant group. */
export default function Settings({ theme, onThemeChange, onModelsChanged }: Props) {
  const [params, setParams] = useSearchParams();
  const requested = params.get('tab');
  const tab: TabId = TABS.find((t) => t.id === requested)?.id ?? 'general';

  return (
    <div className="page">
      <div className={`settings-page${tab === 'models' ? ' settings-page--wide' : ''}`}>
        <h2 style={{ marginTop: 0 }}>Settings</h2>

        <div className="tab-strip" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={`settings-tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls="settings-panel"
              className={`tab-strip__tab${tab === t.id ? ' active' : ''}`}
              onClick={() => setParams({ tab: t.id }, { replace: true })}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div id="settings-panel" role="tabpanel" aria-labelledby={`settings-tab-${tab}`}>
          {tab === 'general' && <GeneralTab theme={theme} onThemeChange={onThemeChange} />}
          {tab === 'comfyui' && <ComfyUITab />}
          {tab === 'models' && <Models onModelsChanged={onModelsChanged} />}
          {tab === 'library' && <LibraryDataTab />}
          {tab === 'integrations' && <IntegrationsTab />}
        </div>
      </div>
    </div>
  );
}
