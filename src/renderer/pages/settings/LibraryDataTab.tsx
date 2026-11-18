import { useEffect, useState } from 'react';
import CleanupSettings from '../../components/CleanupSettings';
import { DbInfo } from '../../../shared/types';
import { formatBytes } from '../../utils/format';
import SettingsSection from './SettingsSection';

export default function LibraryDataTab() {
  const [error, setError] = useState<string | null>(null);
  const [dbInfo, setDbInfo] = useState<DbInfo | null>(null);

  useEffect(() => {
    window.kvgenius.getDbInfo().then(setDbInfo);
  }, []);

  async function runDbAction(action: () => Promise<unknown>) {
    setError(null);
    try {
      // The choose/reset actions relaunch the app on success - nothing more to do here.
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <>
      <CleanupSettings />

      <SettingsSection
        title="Database location"
        description={
          <span style={{ wordBreak: 'break-all' }}>
            Current: {dbInfo?.path ?? '...'} {dbInfo?.isDefault ? '(default)' : ''}
            {dbInfo?.sizeBytes != null ? ` — ${formatBytes(dbInfo.sizeBytes)}` : ''}
          </span>
        }
      >
        {dbInfo && dbInfo.backups.length > 0 && (
          <p className="settings-hint" style={{ wordBreak: 'break-all' }}>
            Copies made before an update changed your data - KVGenius never deletes them, so remove them yourself when you are happy
            everything is fine:
            <br />
            {dbInfo.backups.join(' | ')}
          </p>
        )}
        <div className="button-row">
          <button type="button" onClick={() => runDbAction(() => window.kvgenius.revealDbInFileManager())}>
            Show in File Manager
          </button>
          <button type="button" onClick={() => runDbAction(() => window.kvgenius.chooseExistingDb())}>
            Choose Existing File...
          </button>
          <button type="button" onClick={() => runDbAction(() => window.kvgenius.chooseNewDbLocation())}>
            Choose New Parent Folder...
          </button>
          <button type="button" onClick={() => runDbAction(() => window.kvgenius.resetDbToDefault())}>
            Reset to Default
          </button>
        </div>
        <p className="settings-hint">
          "Choose New Parent Folder..." creates a 'KVGenius_Data' folder inside whatever you pick and puts the database
          and generated images together inside it - so pointing two different apps at the same shared parent folder
          (e.g. a synced backup location) can't mix their files together. Changing the database location restarts
          KVGenius (a running database connection can't be repointed at a new file).
        </p>
        {error && <p className="settings-message settings-message--error">{error}</p>}
      </SettingsSection>
    </>
  );
}
