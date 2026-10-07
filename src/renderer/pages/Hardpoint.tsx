import { useCallback, useEffect, useState } from 'react';
import { HARDPOINT_API_BASE, isHardpointReachable } from '../utils/hardpoint';
import './Hardpoint.css';

/**
 * Embeds Hardpoint's loopback UI when the Hardpoint app is running.
 * Service start/stop/unload live there — KVGenius only deep-links / iframes.
 * The embed is asked to use KVGenius's current theme and drop its own title bar
 * (needs a Hardpoint build with the `theme` / `bare` embed params; older ones ignore them).
 */
export default function HardpointPage({ theme }: { theme: string | null }) {
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
      {reachable === false && (
        <p style={{ color: 'var(--color-accent-red, #c44)', fontSize: 13, marginTop: 0 }}>
          Hardpoint&apos;s embed API is not answering at {HARDPOINT_API_BASE}/api/status.
          The Hardpoint <em>window</em> can still be open (it talks over IPC); embeds need that
          loopback HTTP server. Restart Hardpoint, then click Refresh — or open{' '}
          <code>{HARDPOINT_API_BASE}/api/status</code> in a browser to verify.
        </p>
      )}
      {reachable && theme && (
        <iframe
          className="hardpoint-frame"
          title="Hardpoint"
          src={`${HARDPOINT_API_BASE}/?theme=${encodeURIComponent(theme)}&bare=1`}
          allow="local-network-access; clipboard-read; clipboard-write"
        />
      )}
      <div className="hardpoint-page-footer">
        {notice && (
          <span style={{ fontSize: 13, color: 'var(--color-text-muted)', marginRight: 'auto' }}>
            {notice}
          </span>
        )}
        <div className="hardpoint-page-actions">
          <button type="button" disabled={busy} onClick={() => void refresh()}>
            Refresh
          </button>
          <button type="button" disabled={busy} onClick={() => void handleOpen()}>
            {reachable ? 'Open Hardpoint window' : 'Start Hardpoint'}
          </button>
        </div>
      </div>
    </div>
  );
}
