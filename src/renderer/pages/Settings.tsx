import { useEffect, useState } from 'react';
import { DbInfo } from '../../shared/types';
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
        <div style={{ display: 'flex', gap: 8, maxWidth: 560, alignItems: 'center' }}>
          <input
            id="comfyui-host"
            type="text"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder={defaultHost}
            style={{ flex: 1 }}
          />
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

        <p style={{ color: connectionColor[connection], fontWeight: 600 }}>{connectionLabel[connection]}</p>

        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          Default: {defaultHost || '...'}. Change this if ComfyUI is running on a different host
          or port. ComfyUI Desktop's Settings → Server-Config shows its configured host/port (it
          isn't always the classic default of localhost:8188).
        </p>
      </section>

      <section style={{ marginBottom: 32 }}>
        <h3>Database Location</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12, wordBreak: 'break-all' }}>
          Current: {dbInfo?.path ?? '...'} {dbInfo?.isDefault ? '(default)' : ''}
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button type="button" onClick={handleRevealDb}>
            Show in File Manager
          </button>
          <button type="button" onClick={handleChooseExistingDb}>
            Choose Existing File...
          </button>
          <button type="button" onClick={handleChooseNewDbLocation}>
            Choose New Location...
          </button>
          <button type="button" onClick={handleResetDb}>
            Reset to Default
          </button>
        </div>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
          Changing the database location restarts KVGenius (a running database connection can't
          be repointed at a new file).
        </p>
      </section>

      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
      {status && <p style={{ color: 'var(--color-accent-green)' }}>{status}</p>}
    </div>
  );
}
