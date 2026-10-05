import { useEffect, useState } from 'react';
import { Routes, Route, NavLink, Link, Navigate, useLocation } from 'react-router-dom';
import Generate from './pages/Generate';
import { useGenerationQueue } from './hooks/useGenerationQueue';
import LibraryLayout from './pages/LibraryLayout';
import LibraryOutput from './pages/LibraryOutput';
import LibraryPrompts from './pages/LibraryPrompts';
import LibraryTrash from './pages/LibraryTrash';
import Settings from './pages/Settings';
import Hardpoint from './pages/Hardpoint';
import Timing from './pages/Timing';
import { GenerationRecord, VideoSourceRequest } from '../shared/types';

const CONNECTION_POLL_MS = 15000;
// ComfyUI can take a while to come up after launch (first start, loading models into memory).
const STARTUP_POLL_MS = 2000;
const STARTUP_TIMEOUT_MS = 4 * 60 * 1000;

type ConnectionStatus = 'checking' | 'connected' | 'unreachable' | 'starting';

export default function App() {
  const [recallRecord, setRecallRecord] = useState<GenerationRecord | null>(null);
  const [videoSource, setVideoSource] = useState<VideoSourceRequest | null>(null);
  const [recallPrompt, setRecallPrompt] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('checking');
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [theme, setThemeState] = useState<string | null>(null);
  // One "Show hidden" switch for the whole app (the top bar button), so it need not be flipped on
  // every page. Deliberately not remembered across launches - hidden content starts out hidden.
  const [showHidden, setShowHidden] = useState(false);
  const location = useLocation();
  // One queue for the whole app: Generate and the Library's Upscale both feed it.
  const queue = useGenerationQueue();

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
      // A launch in progress keeps its own 'starting' state until ComfyUI answers or gives up.
      if (!cancelled) setConnection((prev) => (reachable ? 'connected' : prev === 'starting' ? 'starting' : 'unreachable'));
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
    unreachable: '🔴 ComfyUI not reachable - click to launch',
    starting: '🟡 Starting ComfyUI...',
  };

  async function handleLaunchComfyUI() {
    setLaunchError(null);
    setConnection('starting');
    try {
      const result = await window.kvgenius.launchComfyUI();
      if (result.status === 'error') {
        setLaunchError(result.message);
        setConnection('unreachable');
        return;
      }
      if (result.status === 'cancelled') {
        setConnection('unreachable');
        return;
      }
      const deadline = Date.now() + STARTUP_TIMEOUT_MS;
      while (Date.now() < deadline) {
        if (await window.kvgenius.checkComfyUIConnection()) {
          setConnection('connected');
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_MS));
      }
      setLaunchError('ComfyUI was started but is still not answering. Check its window, or the server address in Settings.');
      setConnection('unreachable');
    } catch (err) {
      setLaunchError(err instanceof Error ? err.message : String(err));
      setConnection('unreachable');
    }
  }

  return (
    <div className="app-shell">
      <div className="top-bar">
        <span className="top-bar__title">KVGenius</span>
        <NavLink to="/" end className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Generate
        </NavLink>
        <span className="top-bar__separator" />
        <span className="top-bar__group-label">Library</span>
        <NavLink to="/library/output" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Output
        </NavLink>
        <NavLink to="/library/prompts" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Prompts
        </NavLink>
        <NavLink to="/library/trash" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Trash
        </NavLink>
        <span className="top-bar__separator" />
        <span className="top-bar__group-label">Stats</span>
        <NavLink to="/timing" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Timing
        </NavLink>
        <span className="top-bar__separator" />
        <NavLink to="/hardpoint" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Hardpoint
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Settings
        </NavLink>
        <span className="top-bar__separator" />
        <button
          type="button"
          className={`top-bar__toggle${showHidden ? ' top-bar__toggle--on' : ''}`}
          aria-pressed={showHidden}
          onClick={() => setShowHidden((v) => !v)}
          title="Include items hidden by the hidden-words rule or by hand, in the Library and Prompts. Applies everywhere; resets when the app restarts."
        >
          {showHidden ? '🙈 Showing hidden' : '🙈 Show hidden'}
        </button>
        {launchError && (
          <span className="top-bar__connection-error" title={launchError}>
            {launchError}
          </span>
        )}
        {connection === 'unreachable' || connection === 'starting' ? (
          <button
            type="button"
            className="top-bar__connection"
            disabled={connection === 'starting'}
            onClick={handleLaunchComfyUI}
            title={
              connection === 'starting'
                ? 'Waiting for ComfyUI to come up...'
                : 'Start ComfyUI. The first time you may be asked which program to run (you can change it, or the server address, in Settings).'
            }
          >
            {connectionLabel[connection]}
          </button>
        ) : (
          <Link to="/settings" className="top-bar__connection">
            {connectionLabel[connection]}
          </Link>
        )}
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
          queue={queue}
          recallRecord={recallRecord}
          onRecalled={() => setRecallRecord(null)}
          recallPrompt={recallPrompt}
          onPromptRecalled={() => setRecallPrompt(null)}
          videoSource={videoSource}
          onVideoSourceHandled={() => setVideoSource(null)}
        />
      </div>
      <Routes>
        <Route path="/library" element={<LibraryLayout />}>
          <Route index element={<Navigate to="output" replace />} />
          <Route path="output" element={<LibraryOutput queue={queue} onRecall={setRecallRecord} onImageToVideo={setVideoSource} showHidden={showHidden} />} />
          <Route
            path="prompts"
            element={
              <LibraryPrompts
                queue={queue}
                onRecallPrompt={setRecallPrompt}
                onRecall={setRecallRecord}
                onImageToVideo={setVideoSource}
                showHidden={showHidden}
              />
            }
          />
          <Route path="trash" element={<LibraryTrash />} />
        </Route>
        <Route path="/timing" element={<Timing />} />
        <Route path="/hardpoint" element={<Hardpoint />} />
        <Route path="/settings" element={<Settings theme={theme} onThemeChange={setThemeState} />} />
      </Routes>
    </div>
  );
}
