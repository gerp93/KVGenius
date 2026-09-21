import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SavedPrompt } from '../../shared/types';

interface Props {
  onRecallPrompt: (prompt: string) => void;
}

export default function LibraryPrompts({ onRecallPrompt }: Props) {
  const [savedPrompts, setSavedPrompts] = useState<SavedPrompt[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    window.kvgenius
      .listSavedPrompts()
      .then(setSavedPrompts)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

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

  return (
    <div>
      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
      <h2 style={{ marginTop: 0 }}>Saved Prompts</h2>
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
