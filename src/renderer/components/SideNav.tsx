import { useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
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

/** The app's navigation: a left rail that is either full (labels, sections that fold) or thin (every page
 * its own icon, sections kept as small captions - nothing hides behind a hover). Which one, and which
 * sections are folded, is remembered. */
export default function SideNav({ connection, gpus, launchError, onLaunchComfyUI, showHidden, onToggleShowHidden }: Props) {
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
            <div key={section.id} className="side-nav__section">
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
                    <NavLink key={item.to} to={item.to} end={item.end} className={linkClass} title={item.label}>
                      <span className="side-nav__icon">{item.icon}</span>
                      <span className="side-nav__label">{item.label}</span>
                    </NavLink>
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
