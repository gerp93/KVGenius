import { useEffect, useState } from 'react';
import { McpInfo } from '../../../shared/types';
import SettingsSection from './SettingsSection';

export default function IntegrationsTab() {
  const [mcp, setMcp] = useState<McpInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [snippetCopied, setSnippetCopied] = useState(false);

  useEffect(() => {
    window.kvgenius.getMcpInfo().then(setMcp);
  }, []);

  async function handleToggleMcp(enabled: boolean) {
    setError(null);
    try {
      setMcp(await window.kvgenius.setMcpEnabled(enabled));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleCopySnippet() {
    if (!mcp) return;
    await navigator.clipboard.writeText(mcp.configSnippet);
    setSnippetCopied(true);
    setTimeout(() => setSnippetCopied(false), 2000);
  }

  async function handleChooseFfmpeg() {
    const next = await window.kvgenius.chooseFfmpegPath();
    if (next) setMcp(next);
  }

  async function handleResetFfmpeg() {
    setMcp(await window.kvgenius.resetFfmpegPath());
  }

  return (
    <>
      <SettingsSection
        title="Other apps (MCP)"
        description="Lets an MCP client such as Claude Desktop queue generations, look through your library and stitch videos, using this app. It only listens on this computer and needs a private key that only apps you configure below have. KVGenius must be open for it to work."
      >
        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={mcp?.enabled ?? false}
            disabled={!mcp}
            onChange={(e) => handleToggleMcp(e.target.checked)}
          />
          Allow other apps to control KVGenius (MCP)
        </label>
        {error && <p className="settings-message settings-message--error">{error}</p>}
        {mcp?.enabled && (
          <>
            <p
              style={{
                color: mcp.running ? 'var(--color-accent-green)' : 'var(--color-accent-red)',
                fontWeight: 600,
              }}
            >
              {mcp.running ? `Running on 127.0.0.1:${mcp.port}` : 'Not running - see the error above, or restart KVGenius.'}
            </p>
            <p className="settings-hint">
              Add this to your MCP client's configuration (for Claude Desktop: Settings → Developer → Edit Config), then
              restart the client.
            </p>
            <pre style={{ overflow: 'auto', fontSize: 12, userSelect: 'text' }}>{mcp.configSnippet}</pre>
            <button type="button" onClick={handleCopySnippet}>
              {snippetCopied ? 'Copied' : 'Copy'}
            </button>
          </>
        )}
      </SettingsSection>

      <SettingsSection title="ffmpeg">
        <p
          style={{
            color: mcp?.ffmpeg.available ? 'var(--color-text-muted)' : 'var(--color-accent-red)',
            fontSize: 12,
            wordBreak: 'break-all',
          }}
        >
          {mcp
            ? mcp.ffmpeg.available
              ? `Using ${mcp.ffmpeg.path}${mcp.ffmpeg.override ? ' (chosen here)' : ''}.`
              : 'Not found. Stitching clips into a video and video previews need ffmpeg (with ffprobe next to it).'
            : '...'}
        </p>
        <div className="button-row">
          <button type="button" onClick={handleChooseFfmpeg}>
            Choose ffmpeg...
          </button>
          {mcp?.ffmpeg.override && (
            <button type="button" onClick={handleResetFfmpeg}>
              Reset to Default
            </button>
          )}
        </div>
      </SettingsSection>
    </>
  );
}
