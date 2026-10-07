import { useEffect, useState } from 'react';
import CleanupSettings from '../../components/CleanupSettings';
import { DbInfo } from '../../../shared/types';
import { formatBytes } from '../../utils/format';
import SettingsSection from './SettingsSection';

export default function LibraryDataTab() {
  const [error, setError] = useState<string | null>(null);
  const [dbInfo, setDbInfo] = useState<DbInfo | null>(null);

  // One word or phrase per line; matched as whole words against each prompt (see shared/hiddenWords.ts).
  const [hiddenWordsText, setHiddenWordsText] = useState('');
  const [hiddenBusy, setHiddenBusy] = useState(false);
  const [hiddenMessage, setHiddenMessage] = useState<string | null>(null);

  useEffect(() => {
    window.kvgenius.getDbInfo().then(setDbInfo);
    window.kvgenius.getHiddenWords().then((words) => setHiddenWordsText(words.join('\n')));
  }, []);

  /** The textarea as a word list: one entry per line, commas also accepted. */
  function parseHiddenWords(): string[] {
    return hiddenWordsText.split(/[\n,]/);
  }

  async function saveHiddenWords(): Promise<string[]> {
    const saved = await window.kvgenius.setHiddenWords(parseHiddenWords());
    setHiddenWordsText(saved.join('\n'));
    return saved;
  }

  async function handleSaveHiddenWords() {
    setHiddenBusy(true);
    setHiddenMessage(null);
    try {
      const saved = await saveHiddenWords();
      setHiddenMessage(`Saved ${saved.length} ${saved.length === 1 ? 'word' : 'words'}. New generations use it from now on.`);
    } catch (err) {
      setHiddenMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setHiddenBusy(false);
    }
  }

  async function handleApplyHiddenWords() {
    setHiddenBusy(true);
    setHiddenMessage(null);
    try {
      await saveHiddenWords();
      const { checked, newlyHidden } = await window.kvgenius.applyHiddenWords();
      setHiddenMessage(
        `Checked ${checked} ${checked === 1 ? 'generation' : 'generations'} that weren't hidden; hid ${newlyHidden}.`
      );
    } catch (err) {
      setHiddenMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setHiddenBusy(false);
    }
  }

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
      <SettingsSection
        title="Hidden content"
        description={
          <>
            When a prompt contains any of these words or phrases, its image or video is marked hidden and left out of
            the Library and Prompts unless "Show hidden" (top bar) is on. One per line; matching ignores case and only counts whole
            words.
          </>
        }
      >
        <textarea
          id="hidden-words"
          aria-label="Hidden words"
          value={hiddenWordsText}
          onChange={(e) => setHiddenWordsText(e.target.value)}
          rows={8}
          spellCheck={false}
          placeholder="one word or phrase per line"
          style={{ width: '100%' }}
        />
        <div className="button-row" style={{ marginTop: 8 }}>
          <button type="button" className="primary" onClick={handleSaveHiddenWords} disabled={hiddenBusy}>
            Save
          </button>
          <button type="button" onClick={handleApplyHiddenWords} disabled={hiddenBusy}>
            Save and Apply to Existing
          </button>
        </div>
        <p className="settings-hint">
          "Apply to Existing" checks every earlier image and video against this list and hides the matches. It only
          ever hides - anything you've hidden by hand stays hidden, and nothing is un-hidden - so you can add words and
          run it again whenever you like. Unhide an item from its Details panel in the Library.
        </p>
        {hiddenMessage && <p className="settings-message settings-message--ok">{hiddenMessage}</p>}
      </SettingsSection>

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
