import { useEffect, useState } from 'react';
import { Routes, Route, NavLink, Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import Generate from './pages/Generate';
import QueuePanel from './components/QueuePanel';
import LibraryDetails from './components/LibraryDetails';
import DetailsDock, { DetailsSlotContext } from './components/DetailsDock';
import GalleryLightbox from './components/GalleryLightbox';
import { useGenerationQueue } from './hooks/useGenerationQueue';
import LibraryLayout from './pages/LibraryLayout';
import LibraryOutput from './pages/LibraryOutput';
import LibraryPrompts from './pages/LibraryPrompts';
import LibraryTrash from './pages/LibraryTrash';
import Settings from './pages/Settings';
import Hardpoint from './pages/Hardpoint';
import Timing from './pages/Timing';
import Styles from './pages/Styles';
import Setup from './pages/Setup';
import Models from './pages/Models';
import LibrarySources from './pages/LibrarySources';
import ToolsLayout from './pages/ToolsLayout';
import ToolsUpscale from './pages/ToolsUpscale';
import { FAMILY_KIND, GenerationRecord, VideoSourceRequest } from '../shared/types';
import { UPSCALE_FAMILY } from '../shared/upscale';
import type { UpscaleRecall } from '../shared/upscale';
import { announceGenerationChange, useGenerationChanges } from './utils/generationChanges';
import { pinNotice } from './utils/library';

const CONNECTION_POLL_MS = 15000;
// ComfyUI can take a while to come up after launch (first start, loading models into memory).
const STARTUP_POLL_MS = 2000;
const STARTUP_TIMEOUT_MS = 4 * 60 * 1000;

type ConnectionStatus = 'checking' | 'connected' | 'unreachable' | 'starting';

const QUEUE_COLLAPSED_KEY = 'kvgenius-queue-bar-collapsed';

function loadQueueCollapsed(): boolean {
  try {
    // Folded to the slim bar unless it was last left open.
    return localStorage.getItem(QUEUE_COLLAPSED_KEY) !== '0';
  } catch {
    return true;
  }
}

export default function App() {
  const [recallRecord, setRecallRecord] = useState<GenerationRecord | null>(null);
  const [videoSource, setVideoSource] = useState<VideoSourceRequest | null>(null);
  // An upscale being re-run in Tools > Upscale, and a picture sent there from Library > Sources.
  const [upscaleRecall, setUpscaleRecall] = useState<UpscaleRecall | null>(null);
  const [recallPrompt, setRecallPrompt] = useState<string | null>(null);
  // Bumped when a style is added, edited or deleted, so Generate's Style dropdown reloads.
  const [stylesVersion, setStylesVersion] = useState(0);
  // Bumped when a saved model is added, edited or deleted, so Generate's Model dropdown reloads.
  const [modelsVersion, setModelsVersion] = useState(0);
  const [connection, setConnection] = useState<ConnectionStatus>('checking');
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [theme, setThemeState] = useState<string | null>(null);
  // One "Show hidden" switch for the whole app (the top bar button), so it need not be flipped on
  // every page. Deliberately not remembered across launches - hidden content starts out hidden.
  const [showHidden, setShowHidden] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  // The queue bar is part of the shell (along the bottom), so it is there on every page; open or folded is remembered.
  const [queueCollapsed, setQueueCollapsedState] = useState(loadQueueCollapsed);
  // A short confirmation from an action in the queue bar (pinned, moved to the Trash).
  const [queueNotice, setQueueNotice] = useState<string | null>(null);
  // One queue for the whole app: Generate and the Library's Upscale both feed it.
  const queue = useGenerationQueue();
  // The finished queue result whose details panel is open (the panel docks to the right of any page).
  const [detailsId, setDetailsId] = useState<number | null>(null);
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  // The element the details panels dock into (a full-height column at the right).
  const [detailsSlot, setDetailsSlot] = useState<HTMLElement | null>(null);
  // Looked up in the queue each time, so favorites / pins made in the panel show at once, and the
  // panel closes by itself if the result goes away.
  const detailsRecord = queue.jobs.find((j) => j.record?.id === detailsId)?.record ?? null;
  // Only one details panel at a time: a Library page opening its own closes this one.
  useGenerationChanges((change) => {
    if (change.kind === 'libraryDetailsOpened') setDetailsId(null);
  });

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

  function setQueueCollapsed(collapsed: boolean) {
    setQueueCollapsedState(collapsed);
    try {
      localStorage.setItem(QUEUE_COLLAPSED_KEY, collapsed ? '1' : '0');
    } catch {
      // Not remembered - the panel still works.
    }
  }

  /** A completed result's ★ in the queue panel. Favoriting moves the file, so the queue's own jobs are
   * repointed here and any Library list or Generate form holding the old path is told. */
  async function handleQueueFavorite(record: GenerationRecord) {
    const favorite = !record.favorite;
    try {
      const { imagePath } = await window.kvgenius.setGenerationFavorite(record.id, favorite);
      queue.updateRecord(record.id, { favorite });
      queue.relocateFile(record.id, record.imagePath, imagePath, window.kvgenius.imageUrlFor(imagePath));
      announceGenerationChange({ kind: 'favorite', id: record.id, favorite, oldPath: record.imagePath, imagePath });
    } catch (err) {
      console.error('Could not change the favorite', err);
    }
  }

  /** A completed result's 📌: pins it (or unpins it) as the example of its prompt. */
  async function handleQueuePin(record: GenerationRecord) {
    const pinned = !record.pinned;
    try {
      const { groupSize } = await window.kvgenius.setGenerationPinned(record.id, pinned);
      queue.updateRecord(record.id, { pinned });
      announceGenerationChange({ kind: 'pinned', id: record.id, pinned });
      showQueueNotice(pinned ? pinNotice(groupSize) : null);
    } catch (err) {
      console.error('Could not change the pin', err);
    }
  }

  /** A completed result's 🗑️: moves it to the Trash (no confirmation - it can be restored there). */
  async function handleQueueDelete(record: GenerationRecord) {
    try {
      await window.kvgenius.trashGenerations([record.id], { includeKept: true });
      queue.removeRecord(record.id);
      announceGenerationChange({ kind: 'trashed', id: record.id, imagePath: record.imagePath });
      showQueueNotice('Moved to the Trash - restore it from Library > Trash.');
    } catch (err) {
      console.error('Could not move to the Trash', err);
    }
  }

  function showQueueNotice(text: string | null) {
    setQueueNotice(text);
    if (text) setTimeout(() => setQueueNotice((current) => (current === text ? null : current)), 4000);
  }

  function openQueueDetails(record: GenerationRecord) {
    const closing = detailsId === record.id;
    setDetailsId(closing ? null : record.id);
    if (!closing) announceGenerationChange({ kind: 'queueDetailsOpened' });
  }

  /** The details panel's Hide / Unhide. */
  async function handleQueueHide(record: GenerationRecord) {
    const hidden = !record.hidden;
    try {
      await window.kvgenius.setGenerationHidden(record.id, hidden);
      queue.updateRecord(record.id, { hidden });
      announceGenerationChange({ kind: 'hidden', id: record.id, hidden });
    } catch (err) {
      showQueueNotice(err instanceof Error ? err.message : String(err));
    }
  }

  /** Save as / show in the file manager, from the details panel. */
  async function runFileAction(action: () => Promise<unknown>) {
    try {
      await action();
    } catch (err) {
      showQueueNotice(err instanceof Error ? err.message : String(err));
    }
  }

  function handleQueueImageToVideo(record: GenerationRecord) {
    setVideoSource({ imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  function handleQueueImageToImage(record: GenerationRecord) {
    setVideoSource({ target: 'image', imagePath: record.imagePath, width: record.width, height: record.height });
    navigate('/');
  }

  /** Re-rack, from anywhere. An upscale is re-run from its kept original in Tools > Upscale; everything
   * else opens as a new tab on Generate. */
  function handleRerack(record: GenerationRecord) {
    if (record.modelFamily === UPSCALE_FAMILY && record.sourceImagePath) {
      setUpscaleRecall({ sourcePath: record.sourceImagePath, outputWidth: record.width });
      navigate('/tools/upscale');
      return;
    }
    setRecallRecord(record);
    navigate('/');
  }

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
    <DetailsSlotContext.Provider value={detailsSlot}>
    <div className="app-shell">
      <div className="top-bar">
        <span className="top-bar__title">KVGenius</span>
        <NavLink to="/" end className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Generate
        </NavLink>
        <NavLink to="/styles" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Styles
        </NavLink>
        <span className="top-bar__separator" />
        <span className="top-bar__group-label">Tools</span>
        <NavLink to="/tools/upscale" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Upscale
        </NavLink>
        <span className="top-bar__separator" />
        <span className="top-bar__group-label">Library</span>
        <NavLink to="/library/output" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Output
        </NavLink>
        <NavLink to="/library/prompts" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Prompts
        </NavLink>
        <NavLink to="/library/sources" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Sources
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
        {connection === 'unreachable' && (
          <Link to="/setup" className="top-bar__link" title="Step-by-step: install ComfyUI, add the models, connect">
            Setup guide
          </Link>
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
          <Link to="/settings?tab=comfyui" className="top-bar__connection">
            {connectionLabel[connection]}
          </Link>
        )}
      </div>
      <div className="app-body">
        <div className="app-main">
          <div className="app-content">
            {/* Generate stays mounted across navigation (instead of going through <Routes>) so its
                in-progress prompt/settings survive a trip to Library or Settings and back - only
                hidden via CSS, never unmounted and reset. Library/Settings still mount fresh on each
                visit via <Routes>, which is what keeps Library's list in sync with new generations. */}
            {/* display: contents keeps this wrapper out of the flex box model entirely when visible,
                so Generate's own .page div is still the direct flex child of .app-content, same as
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
                stylesVersion={stylesVersion}
                modelsVersion={modelsVersion}
              />
            </div>
            <Routes>
              <Route path="/tools" element={<ToolsLayout />}>
                <Route index element={<Navigate to="upscale" replace />} />
                <Route
                  path="upscale"
                  element={
                    <ToolsUpscale
                      queue={queue}
                      onShowQueue={() => setQueueCollapsed(false)}
                      recall={upscaleRecall}
                      onRecallHandled={() => setUpscaleRecall(null)}
                    />
                  }
                />
              </Route>
              <Route path="/library" element={<LibraryLayout />}>
                <Route index element={<Navigate to="output" replace />} />
                <Route
                  path="output"
                  element={
                    <LibraryOutput
                      queue={queue}
                      onRecall={handleRerack}
                      onImageToVideo={setVideoSource}
                      showHidden={showHidden}
                      onShowQueue={() => setQueueCollapsed(false)}
                    />
                  }
                />
                <Route
                  path="prompts"
                  element={
                    <LibraryPrompts
                      queue={queue}
                      onRecallPrompt={setRecallPrompt}
                      onRecall={handleRerack}
                      onImageToVideo={setVideoSource}
                      showHidden={showHidden}
                      onShowQueue={() => setQueueCollapsed(false)}
                    />
                  }
                />
                <Route
                  path="sources"
                  element={
                    <LibrarySources
                      queue={queue}
                      onUpscale={(sourcePath) => {
                        setUpscaleRecall({ sourcePath, outputWidth: 0 });
                        navigate('/tools/upscale');
                      }}
                      onMakeVideo={(request) => {
                        setVideoSource(request);
                        navigate('/');
                      }}
                    />
                  }
                />
                <Route path="trash" element={<LibraryTrash />} />
              </Route>
              <Route path="/styles" element={<Styles onChanged={() => setStylesVersion((v) => v + 1)} />} />
              <Route path="/timing" element={<Timing />} />
              <Route path="/hardpoint" element={<Hardpoint />} />
              <Route path="/setup" element={<Setup />} />
              <Route path="/models" element={<Models onModelsChanged={() => setModelsVersion((v) => v + 1)} />} />
              <Route path="/settings" element={<Settings theme={theme} onThemeChange={setThemeState} />} />
            </Routes>
          </div>
          <QueuePanel
            jobs={queue.jobs}
            now={queue.now}
            progressInfo={queue.progressInfo}
            collapsed={queueCollapsed}
            onToggle={() => setQueueCollapsed(!queueCollapsed)}
            onCancelJob={queue.cancelJob}
            onClearQueued={queue.clearQueued}
            onDismissFailed={queue.dismissFailed}
            notice={queueNotice}
            activeDetailsId={detailsRecord ? detailsRecord.id : null}
            onOpenDetails={openQueueDetails}
            onToggleFavorite={handleQueueFavorite}
            onTogglePinned={handleQueuePin}
            onDelete={handleQueueDelete}
            onRerack={handleRerack}
          />
        </div>
        {/* Where the details panels dock (see DetailsDock): full height, right of the page and queue bar. */}
        <div className="details-slot" ref={setDetailsSlot} />
      </div>

        {detailsRecord && (
          <DetailsDock>
            <LibraryDetails
              record={detailsRecord}
              queue={queue}
              onClose={() => setDetailsId(null)}
              onExpand={() => setDetailsExpanded(true)}
              onToggleFavorite={handleQueueFavorite}
              onTogglePinned={handleQueuePin}
              onToggleHidden={handleQueueHide}
              onDelete={handleQueueDelete}
              onRerack={handleRerack}
              onImageToVideo={handleQueueImageToVideo}
              onImageToImage={handleQueueImageToImage}
              onSaveAs={(r) => void runFileAction(() => window.kvgenius.saveGenerationAs(r.imagePath))}
              onReveal={(r) => void runFileAction(() => window.kvgenius.revealGenerationInFileManager(r.imagePath))}
              onUpscaleQueued={() => setQueueCollapsed(false)}
              onGifMade={(made) => announceGenerationChange({ kind: 'created', id: made.id })}
              onError={(message) => message && showQueueNotice(message)}
              onNotice={(message) => message && showQueueNotice(message)}
            />
          </DetailsDock>
        )}
      {detailsExpanded && detailsRecord && (
        <GalleryLightbox
          src={window.kvgenius.imageUrlFor(detailsRecord.imagePath)}
          kind={FAMILY_KIND[detailsRecord.modelFamily] === 'video' ? 'video' : 'image'}
          filePath={detailsRecord.imagePath}
          alt={detailsRecord.prompt}
          hasPrev={false}
          hasNext={false}
          onPrev={() => undefined}
          onNext={() => undefined}
          onClose={() => setDetailsExpanded(false)}
        />
      )}
    </div>
    </DetailsSlotContext.Provider>
  );
}
