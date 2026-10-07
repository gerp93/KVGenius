import { useEffect, useState } from 'react';
import { MAX_STYLE_NAME_LENGTH, MAX_STYLE_TEXT_LENGTH, PromptStyle, combinePrompt } from '../../shared/styles';

interface Props {
  /** Tells the app a style was added, edited or deleted, so Generate's dropdown can reload. */
  onChanged: () => void;
}

/** Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method". */
function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
}

/**
 * Where the user defines their prompt styles: reusable wording for a look ("1930s movie poster, bold
 * lithograph, limited palette") that Generate can add after the prompt. A list on the left, the
 * selected style's editor on the right.
 */
export default function Styles({ onChanged }: Props) {
  const [styles, setStyles] = useState<PromptStyle[] | null>(null);
  // The style being edited, or null while writing a new one.
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [name, setName] = useState('');
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listStyles()
      .then((list) => {
        if (!cancelled) setStyles(list);
      })
      .catch((err) => {
        if (!cancelled) {
          setStyles([]);
          setError(cleanError(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = styles?.find((s) => s.id === selectedId) ?? null;
  const dirty = selected ? name !== selected.name || text !== selected.text : name.trim() !== '' || text.trim() !== '';

  function select(style: PromptStyle | null) {
    setSelectedId(style ? style.id : null);
    setName(style?.name ?? '');
    setText(style?.text ?? '');
    setError(null);
    setNotice(null);
    setConfirmingDelete(false);
  }

  async function handleSave() {
    setError(null);
    setNotice(null);
    try {
      const saved = await window.kvgenius.saveStyle({ name, text }, selectedId);
      const list = await window.kvgenius.listStyles();
      setStyles(list);
      setSelectedId(saved.id);
      setName(saved.name);
      setText(saved.text);
      setNotice('Saved.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  async function handleDelete() {
    if (selectedId === null) return;
    // Deleting is permanent, so it takes a second click (a style is quick to rewrite, so no dialog).
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    try {
      await window.kvgenius.deleteStyle(selectedId);
      setStyles(await window.kvgenius.listStyles());
      select(null);
      setNotice('Deleted. Pictures already made with it keep their prompt.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  return (
    <div className="page styles-page">
      <h2 style={{ marginTop: 0 }}>Styles</h2>
      <p className="styles-page__intro">
        A style is wording for a look - say, <em>1930s movie poster, bold lithograph, limited palette, art-deco lettering</em>.
        Pick one on the Generate page and it is added after your prompt, so the prompt can stay about what is in the picture.
        With no style picked, nothing changes: your prompt is sent exactly as you wrote it.
      </p>

      <div className="styles-page__body">
        <div className="styles-page__list">
          <button type="button" className="primary" onClick={() => select(null)} disabled={selectedId === null && !dirty}>
            + New style
          </button>
          {styles === null ? (
            <p className="styles-page__empty">Loading...</p>
          ) : styles.length === 0 ? (
            <p className="styles-page__empty">No styles yet. Write one on the right.</p>
          ) : (
            <ul>
              {styles.map((style) => (
                <li key={style.id}>
                  <button
                    type="button"
                    className={`styles-page__item${style.id === selectedId ? ' styles-page__item--active' : ''}`}
                    onClick={() => select(style)}
                    title={style.text}
                  >
                    {style.name}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="styles-page__editor">
          <label className="field-label" htmlFor="style-name">
            Name
          </label>
          <input
            id="style-name"
            type="text"
            value={name}
            maxLength={MAX_STYLE_NAME_LENGTH}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. 1930s movie poster"
            style={{ width: '100%', maxWidth: 420 }}
          />

          <label className="field-label" htmlFor="style-text" style={{ marginTop: 12 }}>
            Style text
          </label>
          <textarea
            id="style-text"
            value={text}
            maxLength={MAX_STYLE_TEXT_LENGTH}
            onChange={(e) => setText(e.target.value)}
            placeholder="The words that produce the look, added after your prompt"
            rows={6}
            style={{ width: '100%', resize: 'vertical' }}
          />
          <p className="styles-page__hint">
            Example of what is sent: <code>{combinePrompt('a fox in a snowy forest', text.trim() || 'your style text')}</code>
          </p>

          <div className="button-row">
            <button type="button" className="primary" onClick={() => void handleSave()} disabled={!dirty || !name.trim() || !text.trim()}>
              {selected ? 'Save changes' : 'Save style'}
            </button>
            {selected && (
              <button type="button" onClick={() => void handleDelete()} onBlur={() => setConfirmingDelete(false)}>
                {confirmingDelete ? 'Click again to delete' : 'Delete'}
              </button>
            )}
          </div>
          {error && <p className="styles-page__error">{error}</p>}
          {notice && <p className="styles-page__notice">{notice}</p>}
        </div>
      </div>
    </div>
  );
}
