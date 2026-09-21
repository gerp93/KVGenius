import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationRecord, SavedPrompt } from '../../shared/types';

type Tab = 'image' | 'video';

interface Props {
  onRecall: (record: GenerationRecord) => void;
  onRecallPrompt: (prompt: string) => void;
}

export default function Library({ onRecall, onRecallPrompt }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [savedPrompts, setSavedPrompts] = useState<SavedPrompt[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [tab, setTab] = useState<Tab>('image');
  const navigate = useNavigate();

  useEffect(() => {
    window.kvgenius
      .listGenerations()
      .then(setRecords)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

    window.kvgenius
      .listSavedPrompts()
      .then(setSavedPrompts)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const visibleRecords = records.filter((r) => (FAMILY_KIND[r.modelFamily] ?? 'image') === tab);
  const imageCount = records.length - records.filter((r) => FAMILY_KIND[r.modelFamily] === 'video').length;
  const videoCount = records.length - imageCount;

  function handleTabChange(next: Tab) {
    setTab(next);
    setSelectedIds(new Set());
  }

  function handleClick(record: GenerationRecord) {
    onRecall(record);
    navigate('/');
  }

  function handleUsePrompt(prompt: string) {
    onRecallPrompt(prompt);
    navigate('/');
  }

  async function handleDeleteSavedPrompt(id: number) {
    try {
      await window.kvgenius.deleteSavedPrompt(id);
      setSavedPrompts((prev) => prev.filter((sp) => sp.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function toggleSelected(id: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelectedIds(new Set(visibleRecords.map((r) => r.id)));
  }

  function clearSelection() {
    setSelectedIds(new Set());
  }

  async function handleDelete(record: GenerationRecord) {
    if (!window.confirm('Delete this generation? This removes the file from disk too.')) return;
    try {
      await window.kvgenius.deleteGeneration(record.id, record.imagePath);
      setRecords((prev) => prev.filter((r) => r.id !== record.id));
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(record.id);
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteSelected() {
    const toDelete = visibleRecords.filter((r) => selectedIds.has(r.id));
    if (toDelete.length === 0) return;
    if (!window.confirm(`Delete ${toDelete.length} generation${toDelete.length === 1 ? '' : 's'}? This removes the files from disk too.`)) {
      return;
    }
    try {
      await Promise.all(toDelete.map((r) => window.kvgenius.deleteGeneration(r.id, r.imagePath)));
      const deletedIds = new Set(toDelete.map((r) => r.id));
      setRecords((prev) => prev.filter((r) => !deletedIds.has(r.id)));
      setSelectedIds(new Set());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSaveAs(record: GenerationRecord) {
    try {
      await window.kvgenius.saveGenerationAs(record.imagePath);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleReveal(record: GenerationRecord) {
    try {
      await window.kvgenius.revealGenerationInFileManager(record.imagePath);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="page">
      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

      <h2 style={{ marginTop: 0 }}>Generations</h2>
      <div className="button-row" style={{ marginBottom: 12 }}>
        <button type="button" className={tab === 'image' ? 'primary' : undefined} onClick={() => handleTabChange('image')}>
          🖼️ Images ({imageCount})
        </button>
        <button type="button" className={tab === 'video' ? 'primary' : undefined} onClick={() => handleTabChange('video')}>
          🎬 Videos ({videoCount})
        </button>
      </div>
      {visibleRecords.length === 0 && (
        <p style={{ color: 'var(--color-text-muted)' }}>
          {tab === 'video' ? 'No videos yet - go make something.' : 'No images yet - go make something.'}
        </p>
      )}

      {visibleRecords.length > 0 && (
        <div className="button-row" style={{ marginBottom: 12 }}>
          <button type="button" onClick={selectAll} disabled={selectedIds.size === visibleRecords.length}>
            Select All
          </button>
          <button type="button" onClick={clearSelection} disabled={selectedIds.size === 0}>
            Clear Selection
          </button>
          <button type="button" onClick={handleDeleteSelected} disabled={selectedIds.size === 0}>
            Delete Selected ({selectedIds.size})
          </button>
        </div>
      )}

      <div className="library-grid">
        {visibleRecords.map((record) => (
          <div key={record.id} className="library-card">
            <label className="library-card__select" onClick={(e) => e.stopPropagation()}>
              <input
                type="checkbox"
                checked={selectedIds.has(record.id)}
                onChange={() => toggleSelected(record.id)}
              />
            </label>
            <div onClick={() => handleClick(record)} style={{ cursor: 'pointer' }}>
              {FAMILY_KIND[record.modelFamily] === 'video' ? (
                <div className="library-card__video">
                  {/* First frame as the thumbnail: the #t fragment seeks just past 0 so the
                      browser paints a frame instead of a blank box. */}
                  <video
                    src={`${window.kvgenius.imageUrlFor(record.imagePath)}#t=0.1`}
                    preload="metadata"
                    muted
                    playsInline
                  />
                  <span className="library-card__play-badge">▶</span>
                </div>
              ) : (
                <img src={window.kvgenius.imageUrlFor(record.imagePath)} alt={record.prompt} />
              )}
              <div className="library-card__info" title={record.prompt}>
                {record.prompt}
              </div>
            </div>
            <div className="library-card__actions">
              <button type="button" onClick={() => handleSaveAs(record)} title="Save As...">
                💾
              </button>
              <button type="button" onClick={() => handleReveal(record)} title="Show in File Manager">
                📂
              </button>
              <button type="button" onClick={() => handleDelete(record)} title="Delete">
                🗑️
              </button>
            </div>
          </div>
        ))}
      </div>

      <h2 style={{ marginTop: 32 }}>Saved Prompts</h2>
      {savedPrompts.length === 0 ? (
        <p style={{ color: 'var(--color-text-muted)' }}>
          No saved prompts yet - use "Save Prompt" on the Generate page.
        </p>
      ) : (
        <div className="saved-prompts-list">
          {savedPrompts.map((sp) => (
            <div key={sp.id} className="saved-prompts-list__row">
              <span
                onClick={() => handleUsePrompt(sp.prompt)}
                title="Click to use this prompt"
                className="saved-prompts-list__text"
              >
                {sp.prompt}
              </span>
              <button type="button" onClick={() => handleDeleteSavedPrompt(sp.id)} title="Delete">
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
