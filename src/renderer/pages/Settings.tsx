import { useEffect, useState } from 'react';
import { ComfyUILauncherInfo, DbInfo, McpInfo, UpdateCheckResult } from '../../shared/types';
import { THEME_NAMES, themeDisplayName } from '../../shared/themes';

type ConnectionStatus = 'unknown' | 'checking' | 'connected' | 'unreachable';

interface Props {
  theme: string | null;
  onThemeChange: (theme: string) => void;
}

export default function Settings({ theme, onThemeChange }: Props) {
  const [host, setHost] = useState('');
  const [defaultHost, setDefaultHost] = useState('');
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('unknown');
  const [dbInfo, setDbInfo] = useState<DbInfo | null>(null);
  const [mcp, setMcp] = useState<McpInfo | null>(null);
  const [launcher, setLauncher] = useState<ComfyUILauncherInfo | null>(null);
  const [snippetCopied, setSnippetCopied] = useState(false);

  // One word or phrase per line; matched as whole words against each prompt (see shared/hiddenWords.ts).
  const [hiddenWordsText, setHiddenWordsText] = useState('');
  const [hiddenBusy, setHiddenBusy] = useState(false);
  const [hiddenMessage, setHiddenMessage] = useState<string | null>(null);

  const [appVersion, setAppVersion] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<UpdateCheckResult['status'] | 'idle' | 'checking'>('idle');
  const [updateMessage, setUpdateMessage] = useState<string | null>(null);

  useEffect(() => {
    window.kvgenius
      .getComfyUIHost()
      .then((info) => {
        setHost(info.host);
        setDefaultHost(info.defaultHost);
        return checkConnection();
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    window.kvgenius.getDbInfo().then(setDbInfo);
    window.kvgenius.getMcpInfo().then(setMcp);
    window.kvgenius.getComfyUILauncher().then(setLauncher);
    window.kvgenius.getHiddenWords().then((words) => setHiddenWordsText(words.join('\n')));
    window.kvgenius.getAppVersion().then(setAppVersion);
  }, []);

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

  async function handleChooseLauncher() {
    const next = await window.kvgenius.chooseComfyUILauncher();
    if (next) setLauncher(next);
  }

  async function handleClearLauncher() {
    setLauncher(await window.kvgenius.clearComfyUILauncher());
  }

  async function handleToggleMcp(enabled: boolean) {
    setError(null);
    try {
      setMcp(await window.kvgenius.setMcpEnabled(enabled));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleCopySnippet() {
    if (!mcp) return;
    await navigator.clipboard.writeText(mcp.configSnippet);
    setSnippetCopied(true);
    setTimeout(() => setSnippetCopied(false), 2000);
  }

  async function handleChooseFfmpeg() {
    const next = await window.kvgenius.chooseFfmpegPath();
    if (next) setMcp(next);
  }

  async function handleResetFfmpeg() {
    setMcp(await window.kvgenius.resetFfmpegPath());
  }

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

  async function checkConnection() {
    setConnection('checking');
    const reachable = await window.kvgenius.checkComfyUIConnection();
    setConnection(reachable ? 'connected' : 'unreachable');
  }

  async function handleSave() {
    setError(null);
    try {
      await window.kvgenius.setComfyUIHost(host);
      setStatus('Saved.');
      setTimeout(() => setStatus(null), 2000);
      await checkConnection();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleReset() {
    setError(null);
    try {
      await window.kvgenius.resetComfyUIHost();
      const info = await window.kvgenius.getComfyUIHost();
      setHost(info.host);
      setStatus('Reset to default.');
      setTimeout(() => setStatus(null), 2000);
      await checkConnection();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleThemeChange(newTheme: string) {
    onThemeChange(newTheme);
    try {
      await window.kvgenius.setTheme(newTheme);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRevealDb() {
    setError(null);
    try {
      await window.kvgenius.revealDbInFileManager();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleChooseExistingDb() {
    setError(null);
    try {
      // A successful pick relaunches the app - nothing more to do here.
      await window.kvgenius.chooseExistingDb();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleChooseNewDbLocation() {
    setError(null);
    try {
      await window.kvgenius.chooseNewDbLocation();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleResetDb() {
    setError(null);
    try {
      await window.kvgenius.resetDbToDefault();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const connectionLabel: Record<ConnectionStatus, string> = {
    unknown: '',
    checking: '⏳ Checking...',
    connected: '🟢 Connected',
    unreachable: '🔴 Not reachable',
  };
  const connectionColor: Record<ConnectionStatus, string> = {
    unknown: 'var(--color-text-muted)',
    checking: 'var(--color-text-muted)',
    connected: 'var(--color-accent-green)',
    unreachable: 'var(--color-accent-red)',
  };

  return (
    <div className="page">
      <h2 style={{ marginTop: 0 }}>Settings</h2>

      <section style={{ marginBottom: 32 }}>
        <h3>Theme</h3>
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
      </section>

      <section style={{ marginBottom: 32 }}>
        <h3>ComfyUI Server</h3>
        <label className="field-label" htmlFor="comfyui-host">
          Server address
        </label>
        <div style={{ display: 'flex', gap: 8, maxWidth: 640, alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            id="comfyui-host"
            type="text"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder={defaultHost}
            style={{ flex: 1, minWidth: 200 }}
          />
          <div className="button-row" style={{ flexWrap: 'nowrap' }}>
            <button type="button" className="primary" onClick={handleSave}>
              Save
            </button>
            <button type="button" onClick={handleReset}>
              Reset to Default
            </button>
            <button type="button" onClick={checkConnection}>
              Test Connection
            </button>
          </div>
        </div>

        <p style={{ color: connectionColor[connection], fontWeight: 600 }}>{connectionLabel[connection]}</p>

        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          Default: {defaultHost || '...'} (ComfyUI Desktop's own default - the standalone ComfyUI
          server defaults to port 8188 instead). ComfyUI Desktop's Settings → Server-Config shows
          its actual configured host/port if this doesn't connect.
        </p>

        <h4 style={{ marginTop: 20 }}>Launch shortcut</h4>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          When ComfyUI isn't running, click the red indicator at the top to start it. KVGenius runs
          the program below: ComfyUI Desktop is found automatically if it's in its default location;
          for the standalone version choose its run script (e.g. run_nvidia_gpu.bat) or AppImage.
        </p>
        <p style={{ fontSize: 12, wordBreak: 'break-all' }}>
          {launcher
            ? launcher.configured
              ? `Will run: ${launcher.configured}`
              : launcher.detected
                ? `Will run (found automatically): ${launcher.detected}`
                : 'Not set - you will be asked when you first click the indicator.'
            : '...'}
        </p>
        <div className="button-row">
          <button type="button" onClick={handleChooseLauncher}>
            Choose Program...
          </button>
          {launcher?.configured && (
            <button type="button" onClick={handleClearLauncher}>
              Clear
            </button>
          )}
        </div>
      </section>

      <section style={{ marginBottom: 32 }}>
        <h3>Hidden Content</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          When a prompt contains any of these words or phrases, its image or video is marked hidden and left out of the
          Library and Prompts unless "Show hidden" (top bar) is on. One per line; matching ignores case and only counts whole words.
        </p>
        <textarea
          id="hidden-words"
          value={hiddenWordsText}
          onChange={(e) => setHiddenWordsText(e.target.value)}
          rows={8}
          spellCheck={false}
          placeholder="one word or phrase per line"
          style={{ width: '100%', maxWidth: 640 }}
        />
        <div className="button-row" style={{ marginTop: 8 }}>
          <button type="button" className="primary" onClick={handleSaveHiddenWords} disabled={hiddenBusy}>
            Save
          </button>
          <button type="button" onClick={handleApplyHiddenWords} disabled={hiddenBusy}>
            Save and Apply to Existing
          </button>
        </div>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          "Apply to Existing" checks every earlier image and video against this list and hides the matches. It only ever
          hides - anything you've hidden by hand stays hidden, and nothing is un-hidden - so you can add words and run it
          again whenever you like. Unhide an item from its Details panel in the Library.
        </p>
        {hiddenMessage && <p style={{ color: 'var(--color-accent-green)', fontSize: 13 }}>{hiddenMessage}</p>}
      </section>

      <section style={{ marginBottom: 32 }}>
        <h3>Database Location</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12, wordBreak: 'break-all' }}>
          Current: {dbInfo?.path ?? '...'} {dbInfo?.isDefault ? '(default)' : ''}
        </p>
        <div className="button-row">
          <button type="button" onClick={handleRevealDb}>
            Show in File Manager
          </button>
          <button type="button" onClick={handleChooseExistingDb}>
            Choose Existing File...
          </button>
          <button type="button" onClick={handleChooseNewDbLocation}>
            Choose New Parent Folder...
          </button>
          <button type="button" onClick={handleResetDb}>
            Reset to Default
          </button>
        </div>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          "Choose New Parent Folder..." creates a 'KVGenius_Data' folder inside whatever you pick and
          puts the database and generated images together inside it - so pointing two different
          apps at the same shared parent folder (e.g. a synced backup location) can't mix their
          files together. Changing the database location restarts KVGenius (a running database
          connection can't be repointed at a new file).
        </p>
      </section>

      <section style={{ marginBottom: 32 }}>
        <h3>Other Apps (MCP)</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          Lets an MCP client such as Claude Desktop queue generations, look through your library and
          stitch videos, using this app. It only listens on this computer and needs a private key
          that only apps you configure below have. KVGenius must be open for it to work.
        </p>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={mcp?.enabled ?? false}
            disabled={!mcp}
            onChange={(e) => handleToggleMcp(e.target.checked)}
          />
          Allow other apps to control KVGenius (MCP)
        </label>
        {mcp?.enabled && (
          <>
            <p style={{ color: mcp.running ? 'var(--color-accent-green)' : 'var(--color-accent-red)', fontWeight: 600 }}>
              {mcp.running ? `Running on 127.0.0.1:${mcp.port}` : 'Not running - see the error above, or restart KVGenius.'}
            </p>
            <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
              Add this to your MCP client's configuration (for Claude Desktop: Settings → Developer → Edit
              Config), then restart the client.
            </p>
            <pre style={{ maxWidth: 640, overflow: 'auto', fontSize: 12, userSelect: 'text' }}>{mcp.configSnippet}</pre>
            <button type="button" onClick={handleCopySnippet}>
              {snippetCopied ? 'Copied' : 'Copy'}
            </button>
          </>
        )}
        <h4 style={{ marginTop: 20 }}>ffmpeg</h4>
        <p style={{ color: mcp?.ffmpeg.available ? 'var(--color-text-muted)' : 'var(--color-accent-red)', fontSize: 12, wordBreak: 'break-all' }}>
          {mcp
            ? mcp.ffmpeg.available
              ? `Using ${mcp.ffmpeg.path}${mcp.ffmpeg.override ? ' (chosen here)' : ''}.`
              : 'Not found. Stitching clips into a video and video previews need ffmpeg (with ffprobe next to it).'
            : '...'}
        </p>
        <div className="button-row">
          <button type="button" onClick={handleChooseFfmpeg}>
            Choose ffmpeg...
          </button>
          {mcp?.ffmpeg.override && (
            <button type="button" onClick={handleResetFfmpeg}>
              Reset to Default
            </button>
          )}
        </div>
      </section>

      <section style={{ marginBottom: 32 }}>
        <h3>Updates</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          {appVersion ? `You're running version ${appVersion}.` : 'Loading version...'}
        </p>
        <button
          type="button"
          disabled={updateStatus === 'checking' || updateStatus === 'unsupported'}
          onClick={handleCheckForUpdates}
        >
          {updateStatus === 'checking' ? 'Checking...' : 'Check for Updates'}
        </button>
        {updateStatus === 'not-available' && (
          <p style={{ color: 'var(--color-text-muted)', fontSize: 12, marginTop: 8 }}>You're up to date.</p>
        )}
        {updateStatus === 'available' && (
          <p style={{ color: 'var(--color-accent-green)', fontSize: 12, marginTop: 8 }}>{updateMessage}</p>
        )}
        {updateStatus === 'error' && (
          <p style={{ color: 'var(--color-accent-red)', fontSize: 12, marginTop: 8 }}>Check failed: {updateMessage}</p>
        )}
        {updateStatus === 'unsupported' && (
          <p style={{ color: 'var(--color-text-muted)', fontSize: 12, marginTop: 8 }}>
            Update checks are only available in a packaged build, not in dev mode.
          </p>
        )}
      </section>

      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
      {status && <p style={{ color: 'var(--color-accent-green)' }}>{status}</p>}
    </div>
  );
}
