import { Fragment, useRef, useState } from 'react';
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom';
import type { PromptTabsModel } from '../utils/promptTabs';
import { GpuInfo, formatVram, gpuLabel } from '../../shared/gpuInfo';
import './SideNav.css';

type NavItem = { to: string; icon: string; label: string; end?: boolean };
type NavSection = { id: string; label: string; icon: string; items: NavItem[] };

// Sections hold the pages that belong together; each page has its own icon so the rail stays fully
// usable when thin. Create is everything that makes a picture or video (Styles shapes the wording Generate
// sends); Utilities holds the pages that look at the app rather than make something. Models lives in Settings.
const SECTIONS: NavSection[] = [
  {
    id: 'create',
    label: 'Create',
    icon: '🪄',
    items: [
      { to: '/', icon: '✨', label: 'Image / Video', end: true },
      { to: '/tools/upscale', icon: '🔍', label: 'Upscale' },
      { to: '/styles', icon: '🎨', label: 'Styles' },
    ],
  },
  {
    id: 'library',
    label: 'Library',
    icon: '📚',
    items: [
      { to: '/library/output', icon: '🖼️', label: 'Output' },
      { to: '/library/prompts', icon: '📌', label: 'Prompts' },
      { to: '/library/sources', icon: '📥', label: 'Sources' },
      { to: '/library/trash', icon: '🗑️', label: 'Trash' },
    ],
  },
  {
    id: 'utilities',
    label: 'Utilities',
    icon: '🧰',
    items: [
      { to: '/timing', icon: '⏱️', label: 'Timing' },
      { to: '/hardpoint', icon: '🎯', label: 'Hardpoint' },
    ],
  },
];

const THIN_KEY = 'kvgenius-sidebar-thin';
const FOLDED_KEY = 'kvgenius-sidebar-folded';

function loadThin(): boolean {
  try {
    return localStorage.getItem(THIN_KEY) === '1';
  } catch {
    return false;
  }
}

function loadFolded(): string[] {
  try {
    const raw = localStorage.getItem(FOLDED_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : [];
  } catch {
    return [];
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Not remembered; the sidebar still works.
  }
}

type ConnectionStatus = 'checking' | 'connected' | 'unreachable' | 'starting';

type Props = {
  connection: ConnectionStatus;
  /** The device(s) ComfyUI runs on; empty while it is unreachable. */
  gpus: GpuInfo[];
  /** Generate's prompt tabs, listed under Image / Video; null until Generate has published them. */
  promptTabs: PromptTabsModel | null;
  launchError: string | null;
  onLaunchComfyUI: () => void;
  showHidden: boolean;
  onToggleShowHidden: () => void;
};

const CONNECTION_LABEL: Record<ConnectionStatus, string> = {
  checking: 'Checking ComfyUI...',
  connected: 'ComfyUI connected',
  unreachable: 'ComfyUI not reachable - click to launch',
  starting: 'Starting ComfyUI...',
};

const TAB_ROW = 28;
/** The least height the tab list is squeezed to before it scrolls: three tabs, or fewer if that is all there are. */
const tabsMinHeight = (count: number) => Math.min(count, 3) * TAB_ROW;
/** The Create section's other rows (head, Image / Video, New tab, Upscale, Styles), so the rail knows when to stop squeezing the tabs. */
const CREATE_FIXED_HEIGHT = 28 + 30 + 30 + 30 + 30 + 6;

/** The prompt tabs, nested under Image / Video. This list is the only part of the rail that scrolls (see SideNav.css), so
 * everything else stays in place; ＋ sits below it, outside the scroll, so a new tab is always one click away. */
function PromptTabs({ model, thin, onGenerate }: { model: PromptTabsModel; thin: boolean; onGenerate: boolean }) {
  const navigate = useNavigate();
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const skipBlur = useRef(false);

  function open(id: string) {
    model.select(id);
    navigate('/');
  }

  function commitRename() {
    if (renamingId) model.rename(renamingId, draft);
    setRenamingId(null);
  }

  return (
    <>
      <div className="side-nav__tabs" role="tablist" aria-label="Prompt tabs" style={{ minHeight: tabsMinHeight(model.tabs.length) }}>
        {model.tabs.map((tab) => {
          const selected = tab.id === model.activeId;
          return (
            <div key={tab.id} className={`side-nav__tab${selected ? ' side-nav__tab--selected' : ''}${selected && onGenerate ? ' active' : ''}`}>
              {renamingId === tab.id && !thin ? (
                <input
                  autoFocus
                  className="side-nav__tab-rename"
                  value={draft}
                  maxLength={40}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      commitRename();
                    } else if (e.key === 'Escape') {
                      e.preventDefault();
                      skipBlur.current = true;
                      setRenamingId(null);
                    }
                  }}
                  onBlur={() => {
                    if (skipBlur.current) {
                      skipBlur.current = false;
                      return;
                    }
                    commitRename();
                  }}
                />
              ) : (
                <button
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  className="side-nav__tab-main"
                  onClick={() => open(tab.id)}
                  onDoubleClick={() => {
                    setRenamingId(tab.id);
                    setDraft(tab.label);
                  }}
                  title={`${tab.label} - ${tab.mode} (double-click to rename)`}
                >
                  <span className={`side-nav__tab-mode side-nav__tab-mode--${tab.mode}`}>{tab.mode === 'video' ? 'VID' : 'IMG'}</span>
                  <span className="side-nav__tab-text">{tab.label}</span>
                </button>
              )}
              {model.tabs.length > 1 && !thin && (
                <button type="button" className="side-nav__tab-close" onClick={() => model.close(tab.id)} title="Close this tab" aria-label="Close this tab">
                  ✕
                </button>
              )}
            </div>
          );
        })}
      </div>
      <button
        type="button"
        className="side-nav__tab-add"
        disabled={!model.canAdd}
        onClick={() => {
          model.add();
          navigate('/');
        }}
        title={model.canAdd ? 'New prompt tab' : `Up to ${model.maxTabs} tabs`}
        aria-label="New prompt tab"
      >
        {thin ? '＋' : '＋ New tab'}
      </button>
    </>
  );
}

/** The app's navigation: a left rail that is either full (labels, sections that fold) or thin (every page
 * its own icon, sections kept as small captions - nothing hides behind a hover). Which one, and which
 * sections are folded, is remembered. */
export default function SideNav({ connection, gpus, promptTabs, launchError, onLaunchComfyUI, showHidden, onToggleShowHidden }: Props) {
  const location = useLocation();
  const [thin, setThin] = useState(loadThin);
  const [folded, setFolded] = useState<string[]>(loadFolded);

  function toggleThin() {
    const next = !thin;
    setThin(next);
    remember(THIN_KEY, next ? '1' : '0');
  }

  function toggleSection(id: string) {
    const next = folded.includes(id) ? folded.filter((f) => f !== id) : [...folded, id];
    setFolded(next);
    remember(FOLDED_KEY, JSON.stringify(next));
  }

  const linkClass = ({ isActive }: { isActive: boolean }) => `side-nav__item${isActive ? ' active' : ''}`;

  return (
    <nav className={`side-nav${thin ? ' side-nav--thin' : ''}`} aria-label="Main">
      <div className="side-nav__brand">
        <span className="side-nav__title">KVGenius</span>
        <button
          type="button"
          className="side-nav__toggle"
          onClick={toggleThin}
          aria-label={thin ? 'Expand the sidebar' : 'Collapse the sidebar'}
          title={thin ? 'Expand the sidebar' : 'Collapse the sidebar to icons'}
        >
          {thin ? '»' : '«'}
        </button>
      </div>

      <div className="side-nav__scroll">
        {SECTIONS.map((section) => {
          const isFolded = !thin && folded.includes(section.id);
          const holdsCurrentPage = section.items.some((item) => (item.end ? location.pathname === item.to : location.pathname.startsWith(item.to)));
          return (
            <div
              key={section.id}
              className={`side-nav__section side-nav__section--${section.id}`}
              // Folded, the section is just its head: the floor that keeps the tab list usable must not hold the space open.
              style={section.id === 'create' && promptTabs && !isFolded ? { minHeight: CREATE_FIXED_HEIGHT + tabsMinHeight(promptTabs.tabs.length) } : undefined}
            >
              <button
                type="button"
                className={`side-nav__head${holdsCurrentPage && isFolded ? ' active' : ''}`}
                onClick={() => toggleSection(section.id)}
                aria-expanded={!isFolded}
                // In the thin rail the head is only a caption; the pages under it are always showing.
                disabled={thin}
                title={thin ? undefined : isFolded ? `Show ${section.label}` : `Hide ${section.label}`}
              >
                <span className="side-nav__icon">{section.icon}</span>
                <span className="side-nav__head-label">{section.label}</span>
                <span className={`side-nav__chevron${isFolded ? '' : ' side-nav__chevron--open'}`} aria-hidden="true">
                  ▶
                </span>
              </button>
              {!isFolded && (
                <div className="side-nav__items">
                  {section.items.map((item) => (
                    <Fragment key={item.to}>
                      <NavLink to={item.to} end={item.end} className={linkClass} title={item.label}>
                        <span className="side-nav__icon">{item.icon}</span>
                        <span className="side-nav__label">{item.label}</span>
                      </NavLink>
                      {item.to === '/' && promptTabs && <PromptTabs model={promptTabs} thin={thin} onGenerate={location.pathname === '/'} />}
                    </Fragment>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="side-nav__foot">
        <NavLink to="/settings" className={linkClass} title="Settings">
          <span className="side-nav__icon">⚙️</span>
          <span className="side-nav__label">Settings</span>
        </NavLink>
        {/* Always one click from the setup guide; a red dot while ComfyUI cannot be reached, when it is most needed. */}
        <Link
          to="/setup"
          className={`side-nav__item${location.pathname === '/setup' ? ' active' : ''}${connection === 'unreachable' ? ' side-nav__item--attention' : ''}`}
          title="Setup guide: install ComfyUI, add the models, connect"
        >
          <span className="side-nav__icon">❓</span>
          <span className="side-nav__label">Setup guide</span>
        </Link>
        <div className="side-nav__rule" />
        <button
          type="button"
          className={`side-nav__item${showHidden ? ' side-nav__item--on' : ''}`}
          aria-pressed={showHidden}
          onClick={onToggleShowHidden}
          title="Include items hidden by the hidden-words rule or by hand, in the Library and Prompts. Applies everywhere; resets when the app restarts."
        >
          <span className="side-nav__icon">🙈</span>
          <span className="side-nav__label">{showHidden ? 'Showing hidden' : 'Show hidden'}</span>
        </button>
        <div className="side-nav__rule" />
        {launchError && (
          <span className="side-nav__error" title={launchError}>
            {launchError}
          </span>
        )}
        {connection === 'unreachable' || connection === 'starting' ? (
          <button
            type="button"
            className={`side-nav__status side-nav__status--${connection}`}
            disabled={connection === 'starting'}
            onClick={onLaunchComfyUI}
            title={
              connection === 'starting'
                ? 'Waiting for ComfyUI to come up...'
                : 'Start ComfyUI. The first time you may be asked which program to run (you can change it, or the server address, in Settings).'
            }
          >
            <span className="side-nav__dot" />
            <span className="side-nav__label">{CONNECTION_LABEL[connection]}</span>
          </button>
        ) : (
          <Link to="/settings?tab=comfyui" className={`side-nav__status side-nav__status--${connection}`} title={CONNECTION_LABEL[connection]}>
            <span className="side-nav__dot" />
            <span className="side-nav__label">{CONNECTION_LABEL[connection]}</span>
          </Link>
        )}
        {gpus.length > 0 && (
          <div
            className="side-nav__gpu"
            title={gpus.map((g) => `${g.name}${g.vramTotal > 0 ? ` - ${formatVram(g.vramFree)} free of ${formatVram(g.vramTotal)}` : ''}`).join('\n')}
          >
            <span className="side-nav__icon">🖥️</span>
            <span className="side-nav__label">{gpuLabel(gpus[0])}</span>
          </div>
        )}
      </div>
    </nav>
  );
}
