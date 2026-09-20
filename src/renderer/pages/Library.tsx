import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FAMILY_KIND, GenerationRecord, SavedPrompt } from '../../shared/types';

interface Props {
  onRecall: (record: GenerationRecord) => void;
  onRecallPrompt: (prompt: string) => void;
}

export default function Library({ onRecall, onRecallPrompt }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [savedPrompts, setSavedPrompts] = useState<SavedPrompt[]>([]);
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

  return (
    <div className="page">
      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}

      <h2 style={{ marginTop: 0 }}>Generations</h2>
      {records.length === 0 && (
        <p style={{ color: 'var(--color-text-muted)' }}>No generations yet - go make something.</p>
      )}
      <div className="library-grid">
        {records.map((record) => (
          <div key={record.id} className="library-card" onClick={() => handleClick(record)}>
            {FAMILY_KIND[record.modelFamily] === 'video' ? (
              <div className="library-card__video-placeholder">🎬 Video</div>
            ) : (
              <img src={window.kvgenius.imageUrlFor(record.imagePath)} alt={record.prompt} />
            )}
            <div className="library-card__info" title={record.prompt}>
              {record.prompt}
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
