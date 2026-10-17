import { useEffect, useState } from 'react';
import {
  MAX_SECTION_HINT_LENGTH,
  MAX_SECTION_LABEL_LENGTH,
  MAX_TEMPLATE_NAME_LENGTH,
  MAX_TEMPLATE_SECTIONS,
  PromptTemplate,
  TEMPLATE_SEPARATORS,
  TEMPLATE_SEPARATOR_LABEL,
  TemplateSection,
  TemplateSeparator,
  joinSections,
  newSectionId,
} from '../../shared/promptTemplates';

interface Props {
  /** Tells the app a template was added, edited or deleted, so Generate's dropdown can reload. */
  onChanged: () => void;
}

/** Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method". */
function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
}

/** The sections a brand new template starts with - an example to change, not something to keep. */
function starterSections(): TemplateSection[] {
  const taken: string[] = [];
  return ['Subject', 'Outfit', 'Setting', 'Camera', 'Style'].map((label) => {
    const id = newSectionId(taken);
    taken.push(id);
    return { id, label, hint: '' };
  });
}

function sameSections(a: TemplateSection[], b: TemplateSection[]): boolean {
  return a.length === b.length && a.every((s, i) => s.id === b[i].id && s.label === b[i].label && s.hint === b[i].hint);
}

/**
 * Where the user defines prompt templates: a named list of sections (outfit, camera, style...) that Generate can show as separate text boxes in
 * place of the one prompt box. It only changes how a prompt is written - the boxes are joined into one ordinary prompt before anything is sent.
 */
export default function Templates({ onChanged }: Props) {
  const [templates, setTemplates] = useState<PromptTemplate[] | null>(null);
  // The template being edited, or null while writing a new one.
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [name, setName] = useState('');
  const [sections, setSections] = useState<TemplateSection[]>(starterSections);
  const [separator, setSeparator] = useState<TemplateSeparator>('comma');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listPromptTemplates()
      .then((list) => {
        if (!cancelled) setTemplates(list);
      })
      .catch((err) => {
        if (!cancelled) {
          setTemplates([]);
          setError(cleanError(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = templates?.find((t) => t.id === selectedId) ?? null;
  const dirty = selected
    ? name !== selected.name || separator !== selected.separator || !sameSections(sections, selected.sections)
    : name.trim() !== '';
  const canSave = dirty && name.trim() !== '' && sections.length > 0 && sections.every((s) => s.label.trim() !== '');

  /** Opens a template for editing, or (with null) a blank one with the starter sections. */
  function select(template: PromptTemplate | null) {
    setSelectedId(template ? template.id : null);
    setName(template?.name ?? '');
    setSections(template ? template.sections.map((s) => ({ ...s })) : starterSections());
    setSeparator(template?.separator ?? 'comma');
    setError(null);
    setNotice(null);
    setConfirmingDelete(false);
  }

  function updateSection(id: string, change: Partial<TemplateSection>) {
    setSections((prev) => prev.map((s) => (s.id === id ? { ...s, ...change } : s)));
  }

  function moveSection(index: number, by: -1 | 1) {
    setSections((prev) => {
      const target = index + by;
      if (target < 0 || target >= prev.length) return prev;
      const next = [...prev];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  function addSection() {
    setSections((prev) => [...prev, { id: newSectionId(prev.map((s) => s.id)), label: '', hint: '' }]);
  }

  async function handleSave() {
    setError(null);
    setNotice(null);
    try {
      const saved = await window.kvgenius.savePromptTemplate({ name, sections, separator }, selectedId);
      setTemplates(await window.kvgenius.listPromptTemplates());
      setSelectedId(saved.id);
      setName(saved.name);
      setSections(saved.sections.map((s) => ({ ...s })));
      setSeparator(saved.separator);
      setNotice('Saved.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  async function handleDelete() {
    if (selectedId === null) return;
    // Deleting is permanent, so it takes a second click (a template is quick to rewrite, so no dialog).
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    try {
      await window.kvgenius.deletePromptTemplate(selectedId);
      setTemplates(await window.kvgenius.listPromptTemplates());
      select(null);
      setNotice('Deleted. Tabs that used it keep their prompt in the single box.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  // What a prompt written with this template looks like, using the section names as stand-ins.
  const preview = joinSections(
    { sections, separator },
    Object.fromEntries(sections.filter((s) => s.label.trim() !== '').map((s) => [s.id, s.label.trim().toLowerCase()])),
  );

  return (
    <div className="page styles-page">
      <div>
        <h2 className="styles-page__title">Prompt templates</h2>
        <p className="styles-page__intro">
          A template splits the prompt box into sections you define - an outfit, a camera, a style, whatever you like. Pick one on the Image / Video page
          and each section gets its own text box; they are joined into one ordinary prompt when you generate. With no template, you get the single box.
        </p>
        <div className="styles-page__body">
          <div className="styles-page__main">
            <section className="styles-page__group">
              <div className="styles-page__tabs templates-page__head">
                <h3 className="panel__title">Your templates{templates ? ` (${templates.length})` : ''}</h3>
                <button type="button" className="primary styles-page__new" onClick={() => select(null)} disabled={selectedId === null && !dirty}>
                  + New template
                </button>
              </div>
              {templates === null ? (
                <p className="styles-page__empty">Loading...</p>
              ) : templates.length === 0 ? (
                <p className="styles-page__empty">No templates yet. The panel on the right starts you off with a few sections to change as you like.</p>
              ) : (
                <div className="template-list">
                  {templates.map((template) => (
                    <button
                      key={template.id}
                      type="button"
                      className={`template-card${template.id === selectedId ? ' template-card--active' : ''}`}
                      onClick={() => select(template)}
                    >
                      <span className="template-card__name">{template.name}</span>
                      <span className="template-card__sections">{template.sections.map((s) => s.label).join(' · ')}</span>
                    </button>
                  ))}
                </div>
              )}
            </section>
          </div>

          <aside className="panel styles-page__editor">
            <div className="styles-page__editor-head">
              <h3 className="panel__title">{selected ? `Edit "${selected.name}"` : 'New template'}</h3>
              {dirty && <span className="styles-page__unsaved">Unsaved changes</span>}
            </div>

            <div className="styles-page__field">
              <div className="styles-page__field-head">
                <label className="field-label" htmlFor="template-name">
                  Name
                </label>
                <span className="styles-page__count">
                  {name.length} / {MAX_TEMPLATE_NAME_LENGTH}
                </span>
              </div>
              <input
                id="template-name"
                type="text"
                value={name}
                maxLength={MAX_TEMPLATE_NAME_LENGTH}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Character portrait"
                style={{ width: '100%' }}
              />
            </div>

            <div className="styles-page__field">
              <span className="field-label">Sections, top to bottom</span>
              <ol className="template-sections">
                {sections.map((section, index) => (
                  <li key={section.id} className="template-section-row">
                    <div className="template-section-row__fields">
                      <input
                        type="text"
                        aria-label={`Section ${index + 1} name`}
                        value={section.label}
                        maxLength={MAX_SECTION_LABEL_LENGTH}
                        onChange={(e) => updateSection(section.id, { label: e.target.value })}
                        placeholder="Section name, e.g. Outfit"
                      />
                      <input
                        type="text"
                        aria-label={`Section ${index + 1} hint`}
                        value={section.hint}
                        maxLength={MAX_SECTION_HINT_LENGTH}
                        onChange={(e) => updateSection(section.id, { hint: e.target.value })}
                        placeholder="Hint shown in the empty box (optional)"
                      />
                    </div>
                    <div className="template-section-row__buttons">
                      <button type="button" aria-label="Move up" title="Move up" onClick={() => moveSection(index, -1)} disabled={index === 0}>
                        ↑
                      </button>
                      <button
                        type="button"
                        aria-label="Move down"
                        title="Move down"
                        onClick={() => moveSection(index, 1)}
                        disabled={index === sections.length - 1}
                      >
                        ↓
                      </button>
                      <button
                        type="button"
                        aria-label="Remove section"
                        title="Remove this section"
                        onClick={() => setSections((prev) => prev.filter((s) => s.id !== section.id))}
                        disabled={sections.length === 1}
                      >
                        ✕
                      </button>
                    </div>
                  </li>
                ))}
              </ol>
              <button type="button" onClick={addSection} disabled={sections.length >= MAX_TEMPLATE_SECTIONS}>
                + Add section
              </button>
            </div>

            <div className="styles-page__field">
              <span className="field-label">Join the sections with</span>
              <div className="styles-page__kinds" role="radiogroup" aria-label="Join the sections with">
                {TEMPLATE_SEPARATORS.map((s) => (
                  <label key={s} className="styles-page__kind-option">
                    <input type="radio" name="template-separator" checked={separator === s} onChange={() => setSeparator(s)} /> {TEMPLATE_SEPARATOR_LABEL[s]}
                  </label>
                ))}
              </div>
              <span className="styles-page__count">Empty sections are left out.</span>
            </div>

            <div className="styles-page__preview">
              <span className="field-label">What the prompt looks like</span>
              <code style={{ whiteSpace: 'pre-wrap' }}>{preview || '(add a section)'}</code>
            </div>

            <div className="styles-page__actions">
              <button type="button" className="primary" onClick={() => void handleSave()} disabled={!canSave}>
                {selected ? 'Save changes' : 'Save template'}
              </button>
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
            {error && <span className="styles-page__error">{error}</span>}
            {!error && notice && <span className="styles-page__notice">{notice}</span>}
          </aside>
        </div>
      </div>
    </div>
  );
}
