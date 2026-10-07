import { useEffect, useState } from 'react';
import { ComfyUILauncherInfo } from '../../../shared/types';
import SettingsSection from './SettingsSection';

type ConnectionStatus = 'unknown' | 'checking' | 'connected' | 'unreachable';

const CONNECTION_LABEL: Record<ConnectionStatus, string> = {
  unknown: '',
  checking: '⏳ Checking...',
  connected: '🟢 Connected',
  unreachable: '🔴 Not reachable',
};
const CONNECTION_COLOR: Record<ConnectionStatus, string> = {
  unknown: 'var(--color-text-muted)',
  checking: 'var(--color-text-muted)',
  connected: 'var(--color-accent-green)',
  unreachable: 'var(--color-accent-red)',
};

export default function ComfyUITab() {
  const [host, setHost] = useState('');
  const [defaultHost, setDefaultHost] = useState('');
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('unknown');
  const [launcher, setLauncher] = useState<ComfyUILauncherInfo | null>(null);

  useEffect(() => {
    window.kvgenius
      .getComfyUIHost()
      .then((info) => {
        setHost(info.host);
        setDefaultHost(info.defaultHost);
        return checkConnection();
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    window.kvgenius.getComfyUILauncher().then(setLauncher);
  }, []);

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

  async function handleChooseLauncher() {
    const next = await window.kvgenius.chooseComfyUILauncher();
    if (next) setLauncher(next);
  }

  async function handleClearLauncher() {
    setLauncher(await window.kvgenius.clearComfyUILauncher());
  }

  return (
    <>
      <SettingsSection
        title="Server"
        description={
          <>
            Default: {defaultHost || '...'} (ComfyUI Desktop's own default - the standalone ComfyUI server defaults to
            port 8188 instead). ComfyUI Desktop's Settings → Server-Config shows its actual configured host/port if this
            doesn't connect.
          </>
        }
      >
        <label className="field-label" htmlFor="comfyui-host">
          Server address
        </label>
        <div className="settings-row">
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
        <p style={{ color: CONNECTION_COLOR[connection], fontWeight: 600, marginBottom: 0 }}>
          {CONNECTION_LABEL[connection]}
        </p>
        {error && <p className="settings-message settings-message--error">{error}</p>}
        {status && <p className="settings-message settings-message--ok">{status}</p>}
      </SettingsSection>

      <SettingsSection
        title="Launch shortcut"
        description="When ComfyUI isn't running, click the red indicator at the top to start it. KVGenius runs the program below: ComfyUI Desktop is found automatically if it's in its default location; for the standalone version choose its run script (e.g. run_nvidia_gpu.bat) or AppImage."
      >
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
      </SettingsSection>
    </>
  );
}
