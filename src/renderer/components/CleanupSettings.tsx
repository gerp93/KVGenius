import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CleanupSettings as Settings, TrashStats, normalizeDays } from '../../shared/cleanup';
import { formatBytes } from '../utils/format';

const muted = { color: 'var(--color-text-muted)', fontSize: 12 } as const;

function plural(n: number): string {
  return `${n} item${n === 1 ? '' : 's'}`;
}

/** Settings > Library Cleanup: move items nobody kept to the Trash, empty the Trash, and (only if
 * asked) have both happen by themselves about once a day. */
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

  async function save(patch: { autoEnabled?: boolean; olderThanDays?: number; trashRetentionDays?: number }) {
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

  async function handleMoveToTrash() {
    if (!preview || preview.stats.count === 0) return;
    const { count, bytes } = preview.stats;
    if (!window.confirm(`Move ${plural(count)} (${formatBytes(bytes)}) to the Trash? You can restore them from Library > Trash.`)) return;
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
    if (!window.confirm(`Permanently delete ${plural(trash.count)} (${formatBytes(trash.bytes)}) from the Trash? This cannot be undone.`)) return;
    setBusy(true);
    setError(null);
    try {
      const deleted = await window.kvgenius.emptyTrash();
      setMessage(`Deleted ${plural(deleted)} for good.`);
      refreshTrash();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleAuto(enabled: boolean) {
    if (
      enabled &&
      !window.confirm(
        `Turn on automatic cleanup?\n\nAbout once a day, items older than ${age} days that are not favorites or pinned will be moved to the Trash, ` +
          `and items that have been in the Trash for ${retention} days will be deleted for good. The first run is a day from now.`
      )
    ) {
      return;
    }
    await save({ autoEnabled: enabled, olderThanDays: age, trashRetentionDays: retention });
  }

  return (
    <section style={{ marginBottom: 32 }}>
      <h3>Library Cleanup</h3>
      <p style={muted}>
        Clear out old images and videos nobody kept. Favorites and pinned items are never touched. Nothing runs by itself
        unless you turn on the automatic option below.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
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

      <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span>
          In the Trash: {plural(trash.count)}
          {trash.count > 0 ? ` (${formatBytes(trash.bytes)})` : ''}
        </span>
        <Link to="/library/trash" className="top-bar__link" style={{ padding: '4px 10px' }}>
          Open Trash
        </Link>
        <button type="button" onClick={handleEmptyTrash} disabled={busy || trash.count === 0}>
          Empty Trash
        </button>
      </div>

      <div style={{ marginTop: 20 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            type="checkbox"
            checked={settings?.autoEnabled ?? false}
            disabled={!settings}
            onChange={(e) => void handleToggleAuto(e.target.checked)}
          />
          <span>Clean up automatically, about once a day</span>
        </label>
        <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <label htmlFor="cleanup-retention">Delete from the Trash for good after</label>
          <input
            id="cleanup-retention"
            type="number"
            min={1}
            value={retentionDays}
            onChange={(e) => setRetentionDays(e.target.value)}
            onBlur={commitDays}
            style={{ width: 80 }}
          />
          <span>days</span>
        </div>
        <p style={muted}>
          Off unless you tick it. When on, the same rule as above runs by itself: old unkept items move to the Trash, and items
          that have sat in the Trash for the number of days above are deleted for good. Turning it on starts the clock, so
          nothing moves until a day later.
        </p>
        {settings?.lastAutoRun && (
          <p style={muted}>
            Last automatic run: {new Date(settings.lastAutoRun).toLocaleString()}
            {settings.lastAutoSummary ? ` - ${settings.lastAutoSummary}` : ''}
          </p>
        )}
      </div>

      {message && <p style={{ color: 'var(--color-accent-green)', fontSize: 13 }}>{message}</p>}
      {error && <p style={{ color: 'var(--color-accent-red)', fontSize: 13 }}>{error}</p>}
    </section>
  );
}
