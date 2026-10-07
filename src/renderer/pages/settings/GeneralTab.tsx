import { useEffect, useState } from 'react';
import { UpdateCheckResult } from '../../../shared/types';
import { THEME_NAMES, themeDisplayName } from '../../../shared/themes';
import SettingsSection from './SettingsSection';

interface Props {
  theme: string | null;
  onThemeChange: (theme: string) => void;
}

export default function GeneralTab({ theme, onThemeChange }: Props) {
  const [error, setError] = useState<string | null>(null);
  const [appVersion, setAppVersion] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<UpdateCheckResult['status'] | 'idle' | 'checking'>('idle');
  const [updateMessage, setUpdateMessage] = useState<string | null>(null);

  useEffect(() => {
    window.kvgenius.getAppVersion().then(setAppVersion);
  }, []);

  async function handleThemeChange(newTheme: string) {
    onThemeChange(newTheme);
    try {
      await window.kvgenius.setTheme(newTheme);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleCheckForUpdates() {
    setUpdateStatus('checking');
    setUpdateMessage(null);
    const result = await window.kvgenius.checkForUpdates();
    setUpdateStatus(result.status);
    if (result.status === 'available') {
      setUpdateMessage(`Version ${result.version} is downloading in the background.`);
    } else if (result.status === 'error') {
      setUpdateMessage(result.message ?? 'Something went wrong.');
    }
  }

  return (
    <>
      <SettingsSection title="Appearance">
        <label className="field-label" htmlFor="theme-select">
          Color theme
        </label>
        <select
          id="theme-select"
          value={theme ?? ''}
          onChange={(e) => handleThemeChange(e.target.value)}
          style={{ maxWidth: 240 }}
        >
          {THEME_NAMES.map((id) => (
            <option key={id} value={id}>
              {themeDisplayName(id)}
            </option>
          ))}
        </select>
        {error && <p className="settings-message settings-message--error">{error}</p>}
      </SettingsSection>

      <SettingsSection
        title="About & Updates"
        description={appVersion ? `You're running version ${appVersion}.` : 'Loading version...'}
      >
        <button
          type="button"
          disabled={updateStatus === 'checking' || updateStatus === 'unsupported'}
          onClick={handleCheckForUpdates}
        >
          {updateStatus === 'checking' ? 'Checking...' : 'Check for Updates'}
        </button>
        {updateStatus === 'not-available' && <p className="settings-hint">You're up to date.</p>}
        {updateStatus === 'available' && <p className="settings-message settings-message--ok">{updateMessage}</p>}
        {updateStatus === 'error' && (
          <p className="settings-message settings-message--error">Check failed: {updateMessage}</p>
        )}
        {updateStatus === 'unsupported' && (
          <p className="settings-hint">Update checks are only available in a packaged build, not in dev mode.</p>
        )}
      </SettingsSection>
    </>
  );
}
