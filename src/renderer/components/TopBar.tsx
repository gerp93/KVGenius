import { Link } from 'react-router-dom';
import { GpuInfo, formatVram, gpuLabel } from '../../shared/gpuInfo';
import './TopBar.css';

export type ConnectionStatus = 'checking' | 'connected' | 'unreachable' | 'starting';

type Props = {
  connection: ConnectionStatus;
  /** The device(s) ComfyUI runs on; empty while it is unreachable. */
  gpus: GpuInfo[];
  launchError: string | null;
  onLaunchComfyUI: () => void;
};

const CONNECTION_LABEL: Record<ConnectionStatus, string> = {
  checking: 'Checking ComfyUI...',
  connected: 'ComfyUI connected',
  unreachable: 'ComfyUI not reachable - click to launch',
  starting: 'Starting ComfyUI...',
};

/** The bar across the top of the window: the app's name at the left, and at the right what it is
 * talking to - the ComfyUI connection (a launch button while it cannot be reached) and the graphics card. */
export default function TopBar({ connection, gpus, launchError, onLaunchComfyUI }: Props) {
  const label = CONNECTION_LABEL[connection];
  return (
    <header className="top-bar">
      <span className="top-bar__title">KVGenius</span>
      <div className="top-bar__status-group">
        {launchError && (
          <span className="top-bar__error" title={launchError}>
            {launchError}
          </span>
        )}
        {gpus.length > 0 && (
          <div
            className="top-bar__gpu"
            title={gpus.map((g) => `${g.name}${g.vramTotal > 0 ? ` - ${formatVram(g.vramFree)} free of ${formatVram(g.vramTotal)}` : ''}`).join('\n')}
          >
            <span aria-hidden="true">🖥️</span>
            <span>{gpuLabel(gpus[0])}</span>
          </div>
        )}
        {connection === 'unreachable' || connection === 'starting' ? (
          <button
            type="button"
            className={`top-bar__status top-bar__status--${connection}`}
            disabled={connection === 'starting'}
            onClick={onLaunchComfyUI}
            title={
              connection === 'starting'
                ? 'Waiting for ComfyUI to come up...'
                : 'Start ComfyUI. The first time you may be asked which program to run (you can change it, or the server address, in Settings).'
            }
          >
            <span className="top-bar__dot" />
            <span>{label}</span>
          </button>
        ) : (
          <Link to="/settings?tab=comfyui" className={`top-bar__status top-bar__status--${connection}`} title={label}>
            <span className="top-bar__dot" />
            <span>{label}</span>
          </Link>
        )}
      </div>
    </header>
  );
}
