import { useEffect, useState } from 'react';
import { Routes, Route, NavLink, Link } from 'react-router-dom';
import Generate from './pages/Generate';
import Library from './pages/Library';
import Settings from './pages/Settings';
import { GenerationRecord } from '../shared/types';

const CONNECTION_POLL_MS = 15000;

type ConnectionStatus = 'checking' | 'connected' | 'unreachable';

export default function App() {
  const [recallRecord, setRecallRecord] = useState<GenerationRecord | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('checking');
  const [theme, setThemeState] = useState<string | null>(null);

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
      <Routes>
        <Route path="/" element={<Generate recallRecord={recallRecord} onRecalled={() => setRecallRecord(null)} />} />
        <Route path="/library" element={<Library onRecall={setRecallRecord} />} />
        <Route path="/settings" element={<Settings theme={theme} onThemeChange={setThemeState} />} />
      </Routes>
    </div>
  );
}
