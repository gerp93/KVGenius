import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SavedPrompt } from '../../shared/types';
import PromptModal from '../components/PromptModal';

interface Props {
  onRecallPrompt: (prompt: string) => void;
}

/** What to call a prompt in the list - older ones saved before names were required have none. */
function displayName(sp: SavedPrompt): string {
  return sp.name?.trim() || 'Untitled prompt';
}

export default function LibraryPrompts({ onRecallPrompt }: Props) {
  const [savedPrompts, setSavedPrompts] = useState<SavedPrompt[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [activeTags, setActiveTags] = useState<string[]>([]);
  const [editing, setEditing] = useState<SavedPrompt | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    window.kvgenius
      .listSavedPrompts()
      .then(setSavedPrompts)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  // Every tag in use, most-used first, for the filter bar and the modal's suggestions.
  const tagCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const sp of savedPrompts) for (const tag of sp.tags) counts.set(tag, (counts.get(tag) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [savedPrompts]);

  // A tag filter for a tag that no longer exists (its last prompt was deleted/retagged) would match
  // nothing and be impossible to clear from the bar.
  const filterTags = activeTags.filter((tag) => tagCounts.some(([t]) => t === tag));

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return savedPrompts.filter((sp) => {
      if (!filterTags.every((tag) => sp.tags.includes(tag))) return false;
      if (!needle) return true;
      return (
        displayName(sp).toLowerCase().includes(needle) ||
        sp.prompt.toLowerCase().includes(needle) ||
        sp.tags.some((tag) => tag.toLowerCase().includes(needle))
      );
    });
  }, [savedPrompts, search, filterTags]);

  function handleUsePrompt(prompt: string) {
    onRecallPrompt(prompt);
    navigate('/');
  }

  function toggleTag(tag: string) {
    setActiveTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]));
  }

  async function handleDelete(sp: SavedPrompt) {
    if (!window.confirm(`Delete "${displayName(sp)}"?`)) return;
    try {
      await window.kvgenius.deleteSavedPrompt(sp.id);
      setSavedPrompts((prev) => prev.filter((p) => p.id !== sp.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSaveEdit(name: string, tags: string[]) {
    if (!editing) return;
    const updated = await window.kvgenius.updateSavedPrompt(editing.id, name, tags);
    setSavedPrompts((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
    setEditing(null);
  }

  return (
    <div className="prompt-library">
      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

      {savedPrompts.length === 0 ? (
        <p style={{ color: 'var(--color-text-muted)' }}>
          No saved prompts yet - use "Save Prompt" on the Generate page.
        </p>
      ) : (
        <>
          <div className="prompt-library__toolbar">
            <input
              type="text"
              className="prompt-library__search"
              value={search}
              placeholder="Search names, prompts and tags..."
              onChange={(e) => setSearch(e.target.value)}
            />
            <span className="prompt-library__count">
              {visible.length === savedPrompts.length
                ? `${savedPrompts.length} prompt${savedPrompts.length === 1 ? '' : 's'}`
                : `${visible.length} of ${savedPrompts.length}`}
            </span>
          </div>

          {tagCounts.length > 0 && (
            <div className="prompt-library__tags">
              {tagCounts.map(([tag, count]) => (
                <button
                  key={tag}
                  type="button"
                  className={`tag-filter${filterTags.includes(tag) ? ' tag-filter--on' : ''}`}
                  onClick={() => toggleTag(tag)}
                >
                  {tag} <span className="tag-filter__count">{count}</span>
                </button>
              ))}
              {filterTags.length > 0 && (
                <button type="button" className="tag-filter tag-filter--clear" onClick={() => setActiveTags([])}>
                  Clear filters
                </button>
              )}
            </div>
          )}

          {visible.length === 0 ? (
            <p style={{ color: 'var(--color-text-muted)' }}>No prompts match.</p>
          ) : (
            <div className="prompt-cards">
              {visible.map((sp) => (
                <div key={sp.id} className="prompt-card">
                  <div className="prompt-card__header">
                    <span className={`prompt-card__name${sp.name ? '' : ' prompt-card__name--untitled'}`} title={displayName(sp)}>
                      {displayName(sp)}
                    </span>
                    <span className="prompt-card__actions">
                      <button type="button" className="primary" onClick={() => handleUsePrompt(sp.prompt)} title="Use this prompt on the Generate page">
                        Use
                      </button>
                      <button type="button" onClick={() => setEditing(sp)} title="Rename or change tags">
                        ✎ Edit
                      </button>
                      <button type="button" onClick={() => handleDelete(sp)} title="Delete">
                        🗑️
                      </button>
                    </span>
                  </div>
                  {sp.tags.length > 0 && (
                    <div className="prompt-card__tags">
                      {sp.tags.map((tag) => (
                        <button
                          key={tag}
                          type="button"
                          className={`tag-chip tag-chip--button${filterTags.includes(tag) ? ' tag-chip--on' : ''}`}
                          onClick={() => toggleTag(tag)}
                          title={`Filter by ${tag}`}
                        >
                          {tag}
                        </button>
                      ))}
                    </div>
                  )}
                  <div className="prompt-card__text" onClick={() => handleUsePrompt(sp.prompt)} title="Click to use this prompt">
                    {sp.prompt}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {editing && (
        <PromptModal
          title="Edit saved prompt"
          submitLabel="Save changes"
          prompt={editing.prompt}
          initialName={editing.name ?? ''}
          initialTags={editing.tags}
          existingTags={tagCounts.map(([tag]) => tag)}
          onSave={handleSaveEdit}
          onClose={() => setEditing(null)}
        />
      )}
    </div>
  );
}
