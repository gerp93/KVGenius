import { useEffect, useState } from 'react';
import {
  MAX_STYLE_NAME_LENGTH,
  MAX_STYLE_TEXT_LENGTH,
  PromptStyle,
  STYLE_KINDS,
  STYLE_KIND_LABEL,
  StyleKind,
  combinePrompt,
} from '../../shared/styles';

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
 * Where the user defines their reusable prompt wording, of two kinds: a Style (a general look - "1930s
 * movie poster, bold lithograph"; one per picture) and an Element (a part of the picture used again and
 * again - an outfit, a character; any number). A list on the left, the selected one's editor on the right.
 */
export default function Styles({ onChanged }: Props) {
  const [styles, setStyles] = useState<PromptStyle[] | null>(null);
  // The style being edited, or null while writing a new one.
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [name, setName] = useState('');
  const [text, setText] = useState('');
  const [kind, setKind] = useState<StyleKind>('style');
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
  const dirty = selected ? name !== selected.name || text !== selected.text || kind !== selected.kind : name.trim() !== '' || text.trim() !== '';

  function select(style: PromptStyle | null) {
    setSelectedId(style ? style.id : null);
    setName(style?.name ?? '');
    setText(style?.text ?? '');
    setKind(style?.kind ?? 'style');
    setError(null);
    setNotice(null);
    setConfirmingDelete(false);
  }

  async function handleSave() {
    setError(null);
    setNotice(null);
    try {
      const saved = await window.kvgenius.saveStyle({ name, text, kind }, selectedId);
      const list = await window.kvgenius.listStyles();
      setStyles(list);
      setSelectedId(saved.id);
      setName(saved.name);
      setText(saved.text);
      setKind(saved.kind);
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

  const canSave = dirty && name.trim() !== '' && text.trim() !== '';
  const count = styles?.length ?? 0;
  const label = STYLE_KIND_LABEL[kind].toLowerCase();

  return (
    <div className="page">
      <div className="styles-page">
        <h2 className="styles-page__title">Styles</h2>
        <p className="styles-page__intro">
          A <strong>style</strong> is a general look - <em>1930s movie poster, bold lithograph, limited palette</em> - and a picture
          uses one. An <strong>element</strong> is a part of the picture you use again and again - an outfit, a character - and a picture
          can use any number. Pick them on the Generate page: they are added after your prompt, elements first and the style last. With
          none picked, your prompt is sent exactly as you wrote it.
        </p>

        <div className="styles-page__body">
          <aside className="panel styles-page__list">
            <div className="styles-page__list-head">
              <h3 className="panel__title">Your styles and elements{styles ? ` (${count})` : ''}</h3>
              <button type="button" className="primary" onClick={() => select(null)} disabled={selectedId === null && !dirty}>
                + New
              </button>
            </div>
            {styles === null ? (
              <p className="styles-page__empty">Loading...</p>
            ) : styles.length === 0 ? (
              <p className="styles-page__empty">Nothing yet. Write your first one on the right.</p>
            ) : (
              STYLE_KINDS.map((k) => {
                const items = styles.filter((s) => s.kind === k);
                if (items.length === 0) return null;
                return (
                  <div key={k}>
                    <h4 className="styles-page__section">
                      {STYLE_KIND_LABEL[k]}s ({items.length})
                    </h4>
                    <ul>
                      {items.map((style) => (
                        <li key={style.id}>
                          <button
                            type="button"
                            className={`styles-page__item${style.id === selectedId ? ' styles-page__item--active' : ''}`}
                            onClick={() => select(style)}
                          >
                            <span className="styles-page__item-name">{style.name}</span>
                            <span className="styles-page__item-text">{style.text}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })
            )}
          </aside>

          <section className="panel styles-page__editor">
            <div className="styles-page__editor-head">
              <h3 className="panel__title">{selected ? `Edit "${selected.name}"` : 'New style or element'}</h3>
              {dirty && <span className="styles-page__unsaved">Unsaved changes</span>}
            </div>

            <div className="styles-page__field">
              <span className="field-label">Kind</span>
              <div className="styles-page__kinds" role="radiogroup" aria-label="Kind">
                {STYLE_KINDS.map((k) => (
                  <label key={k} className="styles-page__kind-option">
                    <input type="radio" name="style-kind" checked={kind === k} onChange={() => setKind(k)} />{' '}
                    {STYLE_KIND_LABEL[k]}
                    <span className="styles-page__kind-hint">
                      {k === 'style' ? ' - a general look, one per picture' : ' - an outfit, a character; any number'}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="styles-page__field">
              <div className="styles-page__field-head">
                <label className="field-label" htmlFor="style-name">
                  Name
                </label>
                <span className="styles-page__count">
                  {name.length} / {MAX_STYLE_NAME_LENGTH}
                </span>
              </div>
              <input
                id="style-name"
                type="text"
                value={name}
                maxLength={MAX_STYLE_NAME_LENGTH}
                onChange={(e) => setName(e.target.value)}
                placeholder={kind === 'style' ? 'e.g. 1930s movie poster' : 'e.g. Red trench coat outfit'}
                style={{ width: '100%' }}
              />
            </div>

            <div className="styles-page__field">
              <div className="styles-page__field-head">
                <label className="field-label" htmlFor="style-text">
                  {kind === 'style' ? 'Style text' : 'Element text'}
                </label>
                <span className="styles-page__count">
                  {text.length} / {MAX_STYLE_TEXT_LENGTH}
                </span>
              </div>
              <textarea
                id="style-text"
                value={text}
                maxLength={MAX_STYLE_TEXT_LENGTH}
                onChange={(e) => setText(e.target.value)}
                placeholder={kind === 'style' ? 'The words that produce the look, added after your prompt' : 'The words that describe it, added after your prompt'}
                rows={6}
                style={{ width: '100%', resize: 'vertical' }}
              />
            </div>

            <div className="styles-page__preview">
              <span className="field-label">What is sent</span>
              <code>{combinePrompt('a fox in a snowy forest', text.trim() || `your ${label} text`)}</code>
            </div>

            <div className="styles-page__actions">
              <button type="button" className="primary" onClick={() => void handleSave()} disabled={!canSave}>
                {selected ? 'Save changes' : `Save ${label}`}
              </button>
              {error && <span className="styles-page__error">{error}</span>}
              {!error && notice && <span className="styles-page__notice">{notice}</span>}
              {selected && (
                <button
                  type="button"
                  className={`button-danger${confirmingDelete ? ' button-danger--armed' : ''}`}
                  onClick={() => void handleDelete()}
                  onBlur={() => setConfirmingDelete(false)}
                >
                  {confirmingDelete ? 'Click again to delete' : 'Delete'}
                </button>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
