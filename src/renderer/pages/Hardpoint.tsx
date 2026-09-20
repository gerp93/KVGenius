import { useCallback, useEffect, useState } from 'react';
import { HARDPOINT_API_BASE, isHardpointReachable } from '../utils/hardpoint';
import './Hardpoint.css';

/**
 * Embeds Hardpoint's loopback UI when the Hardpoint app is running.
 * Service start/stop/unload live there — KVGenius only deep-links / iframes.
 */
export default function HardpointPage() {
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setReachable(await isHardpointReachable());
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 3_000);
    return () => window.clearInterval(id);
  }, [refresh]);

  async function handleOpen() {
    setBusy(true);
    setNotice(null);
    try {
      const result = await window.kvgenius.openHardpoint();
      if (result.status === 'error') {
        setNotice(result.message);
      } else {
        setNotice('Starting Hardpoint…');
        for (let i = 0; i < 15; i++) {
          await new Promise((r) => setTimeout(r, 1_000));
          if (await isHardpointReachable()) {
            setReachable(true);
            setNotice(null);
            return;
          }
        }
        setNotice(
          'Hardpoint did not become reachable. Open the hArdpoInt repo and run npm run dev.'
        );
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="hardpoint-page">
      <div className="hardpoint-page-header">
        <div>
          <h1 style={{ fontSize: 18, margin: 0 }}>Hardpoint</h1>
          <p style={{ marginTop: 6, marginBottom: 0, fontSize: 13, color: 'var(--color-text-muted)' }}>
            GPU and loaded models, plus Start / Stop for local AI services, live in Hardpoint —
            the shared dashboard. KVGenius only embeds it here.
          </p>
        </div>
        <div className="hardpoint-page-actions">
          <button type="button" disabled={busy} onClick={() => void refresh()}>
            Refresh
          </button>
          <button type="button" disabled={busy} onClick={() => void handleOpen()}>
            {reachable ? 'Open Hardpoint window' : 'Start Hardpoint'}
          </button>
        </div>
      </div>
      {notice && (
        <p style={{ fontSize: 13, marginTop: 8, color: 'var(--color-text-muted)' }}>
          {notice}
        </p>
      )}
      {reachable === false && (
        <p style={{ color: 'var(--color-accent-red, #c44)', fontSize: 13, marginTop: 12 }}>
          Hardpoint is not running on {HARDPOINT_API_BASE}. Start it to manage services from this
          panel.
        </p>
      )}
      {reachable && (
        <iframe
          className="hardpoint-frame"
          title="Hardpoint"
          src={`${HARDPOINT_API_BASE}/`}
          allow="clipboard-read; clipboard-write"
        />
      )}
    </div>
  );
}
