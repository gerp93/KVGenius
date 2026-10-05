import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CleanupSettings as Settings, TrashStats, normalizeDays } from '../../shared/cleanup';
import { formatBytes } from '../utils/format';

const muted = { color: 'var(--color-text-muted)', fontSize: 12 } as const;
const row = { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' } as const;

function plural(n: number): string {
  return `${n} item${n === 1 ? '' : 's'}`;
}

type Patch = Parameters<typeof window.kvgenius.setCleanupSettings>[0];

/** Settings > Library Cleanup. Deleting is two steps: items go to the Trash (here, or by the Delete
 * button anywhere in the app), and emptying the Trash sends the files to the Recycle Bin. Each step
 * can also run by itself about once a day, but only if its own option is ticked (both start off). */
export default function CleanupSettings() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [ageDays, setAgeDays] = useState('30');
  const [retentionDays, setRetentionDays] = useState('30');
  const [preview, setPreview] = useState<{ days: number; stats: TrashStats } | null>(null);
  const [trash, setTrash] = useState<TrashStats>({ count: 0, bytes: 0 });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));

  const refreshTrash = useCallback(() => {
    window.kvgenius.getTrashStats().then(setTrash).catch(fail);
  }, []);

  useEffect(() => {
    window.kvgenius
      .getCleanupSettings()
      .then((loaded) => {
        setSettings(loaded);
        setAgeDays(String(loaded.olderThanDays));
        setRetentionDays(String(loaded.trashRetentionDays));
      })
      .catch(fail);
    refreshTrash();
  }, [refreshTrash]);

  const age = normalizeDays(ageDays, settings?.olderThanDays ?? 30);
  const retention = normalizeDays(retentionDays, settings?.trashRetentionDays ?? 30);

  async function save(patch: Patch) {
    try {
      setSettings(await window.kvgenius.setCleanupSettings(patch));
    } catch (err) {
      fail(err);
    }
  }

  /** Writes the thresholds back when the user leaves a field (and shows the cleaned-up value). */
  function commitDays() {
    setAgeDays(String(age));
    setRetentionDays(String(retention));
    setPreview(null);
    if (settings && (settings.olderThanDays !== age || settings.trashRetentionDays !== retention)) {
      void save({ olderThanDays: age, trashRetentionDays: retention });
    }
  }

  async function handlePreview() {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      setPreview({ days: age, stats: await window.kvgenius.previewCleanup(age) });
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  // The preview is the check - it showed exactly what will move - and everything moved can be restored.
  async function handleMoveToTrash() {
    if (!preview || preview.stats.count === 0) return;
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.runCleanup(preview.days);
      setMessage(
        `Moved ${plural(result.moved)} to the Trash.` + (result.failed > 0 ? ` ${plural(result.failed)} could not be moved.` : '')
      );
      setPreview(null);
      refreshTrash();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleEmptyTrash() {
    if (trash.count === 0) return;
    if (
      !window.confirm(
        `Send ${plural(trash.count)} (${formatBytes(trash.bytes)}) from the Trash to the Recycle Bin?\n\nThey can no longer be restored into the app, ` +
          `but you can still get the files back from the Recycle Bin.`
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await window.kvgenius.emptyTrash();
      setMessage(
        `Sent ${plural(result.deleted)} to the Recycle Bin.` + (result.failed > 0 ? ` ${plural(result.failed)} could not be sent and stay in the Trash.` : '')
      );
      refreshTrash();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section style={{ marginBottom: 32 }}>
      <h3>Library Cleanup</h3>
      <p style={muted}>
        Deleting is two steps. Deleting an item anywhere in the app (or the cleanup below) only moves it to the Trash, so it
        can always be restored. Emptying the Trash sends the files to your computer's Recycle Bin. Favorites and pinned items
        are never cleaned up automatically. Nothing below runs by itself unless you tick its option.
      </p>

      <h4 style={{ margin: '16px 0 6px' }}>1. Move old items to the Trash</h4>
      <div style={row}>
        <label htmlFor="cleanup-age">Items older than</label>
        <input
          id="cleanup-age"
          type="number"
          min={1}
          value={ageDays}
          onChange={(e) => setAgeDays(e.target.value)}
          onBlur={commitDays}
          style={{ width: 80 }}
        />
        <span>days that are not favorited or pinned</span>
        <button type="button" onClick={handlePreview} disabled={busy}>
          Preview
        </button>
      </div>

      {preview && (
        <div style={{ marginTop: 8 }}>
          <p style={{ margin: '0 0 8px' }}>
            {preview.stats.count === 0
              ? `Nothing is older than ${preview.days} days and unkept.`
              : `${plural(preview.stats.count)} (${formatBytes(preview.stats.bytes)}) would move to the Trash.`}
          </p>
          {preview.stats.count > 0 && (
            <button type="button" className="primary" onClick={handleMoveToTrash} disabled={busy}>
              Move to Trash
            </button>
          )}
        </div>
      )}

      <label style={{ ...row, marginTop: 10 }}>
        <input
          type="checkbox"
          checked={settings?.autoTrashEnabled ?? false}
          disabled={!settings}
          onChange={(e) => void save({ autoTrashEnabled: e.target.checked, olderThanDays: age, trashRetentionDays: retention })}
        />
        <span>Do this automatically, about once a day</span>
      </label>
      <p style={muted}>
        Off unless ticked. Turning it on starts the clock, so nothing moves until a day later.
        {settings?.lastAutoTrashRun
          ? ` Last run: ${new Date(settings.lastAutoTrashRun).toLocaleString()}${settings.lastAutoTrashSummary ? ` - ${settings.lastAutoTrashSummary}` : ''}`
          : ''}
      </p>

      <h4 style={{ margin: '20px 0 6px' }}>2. Empty the Trash</h4>
      <div style={row}>
        <span>
          In the Trash: {plural(trash.count)}
          {trash.count > 0 ? ` (${formatBytes(trash.bytes)})` : ''}
        </span>
        <Link to="/library/trash" className="top-bar__link" style={{ padding: '4px 10px' }}>
          Open Trash
        </Link>
        <button type="button" onClick={handleEmptyTrash} disabled={busy || trash.count === 0}>
          Empty Trash (to the Recycle Bin)
        </button>
      </div>

      <label style={{ ...row, marginTop: 10 }}>
        <input
          type="checkbox"
          checked={settings?.autoEmptyEnabled ?? false}
          disabled={!settings}
          onChange={(e) => void save({ autoEmptyEnabled: e.target.checked, olderThanDays: age, trashRetentionDays: retention })}
        />
        <span>Empty it automatically, about once a day: send items that have been in the Trash for</span>
        <input
          id="cleanup-retention"
          type="number"
          min={1}
          value={retentionDays}
          onChange={(e) => setRetentionDays(e.target.value)}
          onBlur={commitDays}
          style={{ width: 80 }}
        />
        <span>days to the Recycle Bin</span>
      </label>
      <p style={muted}>
        Off unless ticked, and separate from step 1. Turning it on starts the clock, so nothing is sent until a day later.
        {settings?.lastAutoEmptyRun
          ? ` Last run: ${new Date(settings.lastAutoEmptyRun).toLocaleString()}${settings.lastAutoEmptySummary ? ` - ${settings.lastAutoEmptySummary}` : ''}`
          : ''}
      </p>

      {message && <p style={{ color: 'var(--color-accent-green)', fontSize: 13 }}>{message}</p>}
      {error && <p style={{ color: 'var(--color-accent-red)', fontSize: 13 }}>{error}</p>}
    </section>
  );
}
