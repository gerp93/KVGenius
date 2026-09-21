import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { MAX_NAME_LENGTH, MAX_TAGS, MAX_TAG_LENGTH, normalizeName, normalizeTags } from '../../shared/promptTags';

interface Props {
  title: string;
  submitLabel: string;
  /** The prompt being saved, shown read-only so it is clear what the name is for. */
  prompt: string;
  initialName?: string;
  initialTags?: string[];
  /** Tags already used by other prompts, offered as suggestions. */
  existingTags: string[];
  onSave: (name: string, tags: string[]) => Promise<void>;
  onClose: () => void;
}

/** Asks for a name (required) and tags (optional) for a prompt - used to save a new prompt and to
 * edit a saved one. */
export default function PromptModal({
  title,
  submitLabel,
  prompt,
  initialName = '',
  initialTags = [],
  existingTags,
  onSave,
  onClose,
}: Props) {
  const [name, setName] = useState(initialName);
  const [tags, setTags] = useState<string[]>(normalizeTags(initialTags));
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const tagRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    nameRef.current?.focus();
    nameRef.current?.select();
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !saving) onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, saving]);

  const canSave = normalizeName(name) !== '' && !saving;

  function commitDraft(): string[] {
    const next = normalizeTags([...tags, draft]);
    setTags(next);
    setDraft('');
    return next;
  }

  async function submit() {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      // Whatever is typed in the tag box but not yet turned into a chip still counts.
      await onSave(normalizeName(name), normalizeTags([...tags, draft]));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  }

  const suggestions = existingTags.filter((t) => !tags.some((chosen) => chosen.toLowerCase() === t.toLowerCase()));

  return createPortal(
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={title}>
      <form
        className="modal"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            void submit();
          }
        }}
      >
        <h3 className="modal__title">{title}</h3>

        <label className="field-label" htmlFor="prompt-name">
          Name <span className="modal__required">(required)</span>
        </label>
        <input
          id="prompt-name"
          ref={nameRef}
          type="text"
          value={name}
          maxLength={MAX_NAME_LENGTH}
          placeholder="e.g. Golden retriever, studio light"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
              e.preventDefault();
              tagRef.current?.focus();
            }
          }}
          style={{ width: '100%' }}
        />

        <label className="field-label" htmlFor="prompt-tags" style={{ marginTop: 14 }}>
          Tags <span className="modal__optional">(optional - press Enter or comma to add)</span>
        </label>
        <div className="tag-input" onClick={() => tagRef.current?.focus()}>
          {tags.map((tag) => (
            <span key={tag} className="tag-chip">
              {tag}
              <button
                type="button"
                className="tag-chip__remove"
                title={`Remove ${tag}`}
                onClick={() => setTags((prev) => prev.filter((t) => t !== tag))}
              >
                ✕
              </button>
            </span>
          ))}
          <input
            id="prompt-tags"
            ref={tagRef}
            type="text"
            list="prompt-tag-options"
            value={draft}
            maxLength={MAX_TAG_LENGTH}
            disabled={tags.length >= MAX_TAGS}
            placeholder={tags.length >= MAX_TAGS ? `Up to ${MAX_TAGS} tags` : tags.length === 0 ? 'Add a tag...' : ''}
            onChange={(e) => {
              const value = e.target.value;
              if (value.includes(',')) {
                setTags((prev) => normalizeTags([...prev, ...value.split(',')]));
                setDraft('');
              } else {
                setDraft(value);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                commitDraft();
              } else if (e.key === 'Backspace' && draft === '' && tags.length > 0) {
                setTags((prev) => prev.slice(0, -1));
              }
            }}
            onBlur={() => {
              if (draft.trim()) commitDraft();
            }}
          />
          <datalist id="prompt-tag-options">
            {suggestions.map((tag) => (
              <option key={tag} value={tag} />
            ))}
          </datalist>
        </div>

        <div className="field-label" style={{ marginTop: 14 }}>
          Prompt
        </div>
        <div className="modal__prompt">{prompt}</div>

        {error && <p className="modal__error">{error}</p>}

        <div className="modal__actions">
          <button type="button" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={!canSave}>
            {saving ? 'Saving...' : submitLabel}
          </button>
        </div>
      </form>
    </div>,
    document.body
  );
}
