import { useEffect, useState } from 'react';
import { Routes, Route, NavLink, Link, useLocation } from 'react-router-dom';
import Generate from './pages/Generate';
import Library from './pages/Library';
import Settings from './pages/Settings';
import { GenerationRecord } from '../shared/types';

const CONNECTION_POLL_MS = 15000;

type ConnectionStatus = 'checking' | 'connected' | 'unreachable';

export default function App() {
  const [recallRecord, setRecallRecord] = useState<GenerationRecord | null>(null);
  const [recallPrompt, setRecallPrompt] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('checking');
  const [theme, setThemeState] = useState<string | null>(null);
  const location = useLocation();

  useEffect(() => {
    window.kvgenius.getTheme().then(setThemeState);
  }, []);

  useEffect(() => {
    if (theme) document.body.className = `${theme}-theme`;
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      const reachable = await window.kvgenius.checkComfyUIConnection();
      if (!cancelled) setConnection(reachable ? 'connected' : 'unreachable');
    }
    check();
    const interval = setInterval(check, CONNECTION_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const connectionLabel: Record<ConnectionStatus, string> = {
    checking: '⏳ Checking ComfyUI...',
    connected: '🟢 ComfyUI connected',
    unreachable: '🔴 ComfyUI not reachable',
  };

  return (
    <div className="app-shell">
      <div className="top-bar">
        <span className="top-bar__title">KVGenius</span>
        <NavLink to="/" end className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Generate
        </NavLink>
        <NavLink to="/library" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Library
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Settings
        </NavLink>
        <Link
          to="/settings"
          className="top-bar__connection"
          title={connection === 'unreachable' ? 'Click to configure the ComfyUI server address' : undefined}
        >
          {connectionLabel[connection]}
        </Link>
      </div>
      {/* Generate stays mounted across navigation (instead of going through <Routes>) so its
          in-progress prompt/settings survive a trip to Library or Settings and back - only
          hidden via CSS, never unmounted and reset. Library/Settings still mount fresh on each
          visit via <Routes>, which is what keeps Library's list in sync with new generations. */}
      {/* display: contents keeps this wrapper out of the flex box model entirely when visible,
          so Generate's own .page div is still the direct flex child of .app-shell, same as
          when it rendered through <Routes> - needed for its flex: 1 height to keep working. */}
      <div style={{ display: location.pathname === '/' ? 'contents' : 'none' }}>
        <Generate
          recallRecord={recallRecord}
          onRecalled={() => setRecallRecord(null)}
          recallPrompt={recallPrompt}
          onPromptRecalled={() => setRecallPrompt(null)}
        />
      </div>
      <Routes>
        <Route path="/library" element={<Library onRecall={setRecallRecord} onRecallPrompt={setRecallPrompt} />} />
        <Route path="/settings" element={<Settings theme={theme} onThemeChange={setThemeState} />} />
      </Routes>
    </div>
  );
}
