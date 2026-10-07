import { useEffect, useState } from 'react';
import { PROFILE_FAMILIES, SAMPLER_LIMITS, profileFamily } from '../../shared/modelFamilies';
import { MAX_PROFILE_NAME_LENGTH, ModelProfile, SamplerSettings } from '../../shared/modelProfiles';
import { ModelStatusReport } from '../../shared/modelStatus';

interface Props {
  /** What ComfyUI (or the models folder) has, to choose files from. null while the first check runs. */
  report: ModelStatusReport | null;
  /** Tells the app a model was added, edited or deleted, so Generate's dropdown can reload. */
  onChanged: () => void;
}

/** Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method". */
function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
}

const DISTILLED_NAME = /turbo|lightning|lcm|distill|hyper/i;

interface Draft {
  name: string;
  family: string;
  files: Record<string, string>;
  steps: string;
  cfg: string;
  shift: string;
  sampler: string;
  scheduler: string;
}

function newDraft(family = PROFILE_FAMILIES[0].family): Draft {
  const def = profileFamily(family) ?? PROFILE_FAMILIES[0];
  const s = def.sampler as SamplerSettings;
  return {
    name: '',
    family: def.family,
    files: Object.fromEntries(def.slots.map((slot) => [slot.key, slot.defaultFile])),
    steps: String(s.steps),
    cfg: String(s.cfg),
    shift: String(s.shift),
    sampler: s.sampler,
    scheduler: s.scheduler,
  };
}

function draftFrom(profile: ModelProfile): Draft {
  return {
    name: profile.name,
    family: profile.family,
    files: { ...profile.files },
    steps: String(profile.sampler.steps),
    cfg: String(profile.sampler.cfg),
    shift: String(profile.sampler.shift),
    sampler: profile.sampler.sampler,
    scheduler: profile.sampler.scheduler,
  };
}

function sameDraft(a: Draft, b: Draft): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Where the user adds other models of a family the app already supports (a fine-tune, a different checkpoint):
 * which file goes in each loader slot, and the steps / CFG / sampler that suit it. A list on the left, the
 * selected model's editor on the right - the same layout as Styles. The built-in model is always there and
 * is not edited.
 */
export default function ModelProfiles({ report, onChanged }: Props) {
  const [profiles, setProfiles] = useState<ModelProfile[] | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [draft, setDraft] = useState<Draft>(newDraft());
  const [saved, setSaved] = useState<Draft>(newDraft());
  const [choices, setChoices] = useState<{ samplers: string[]; schedulers: string[] }>({ samplers: [], schedulers: [] });
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listModelProfiles()
      .then((list) => !cancelled && setProfiles(list))
      .catch((err) => {
        if (!cancelled) {
          setProfiles([]);
          setError(cleanError(err));
        }
      });
    window.kvgenius.getSamplerChoices().then((c) => !cancelled && setChoices(c)).catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const def = profileFamily(draft.family) ?? PROFILE_FAMILIES[0];
  const selected = profiles?.find((p) => p.id === selectedId) ?? null;
  const dirty = !sameDraft(draft, saved);
  const installed = report && report.source !== 'none' ? report.installed : null;
  const builtInFile = def.slots[0].defaultFile;
  const diffusionFile = draft.files[def.slots[0].key] ?? '';

  function select(profile: ModelProfile | null) {
    const next = profile ? draftFrom(profile) : newDraft();
    setSelectedId(profile ? profile.id : null);
    setDraft(next);
    setSaved(next);
    setError(null);
    setNotice(null);
    setConfirmingDelete(false);
  }

  function setField<K extends keyof Draft>(key: K, value: Draft[K]) {
    setDraft((d) => ({ ...d, [key]: value }));
  }

  function setFile(slotKey: string, file: string) {
    setDraft((d) => ({ ...d, files: { ...d.files, [slotKey]: file } }));
  }

  async function handleSave() {
    setError(null);
    setNotice(null);
    try {
      const result = await window.kvgenius.saveModelProfile(
        {
          family: draft.family,
          name: draft.name,
          files: draft.files,
          sampler: { steps: Number(draft.steps), cfg: Number(draft.cfg), shift: Number(draft.shift), sampler: draft.sampler, scheduler: draft.scheduler },
        },
        selectedId,
      );
      setProfiles(await window.kvgenius.listModelProfiles());
      setSelectedId(result.id);
      const next = draftFrom(result);
      setDraft(next);
      setSaved(next);
      setNotice('Saved.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  async function handleDelete() {
    if (selectedId === null) return;
    // Permanent, so it takes a second click (a model is quick to set up again, so no dialog).
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    try {
      await window.kvgenius.deleteModelProfile(selectedId);
      setProfiles(await window.kvgenius.listModelProfiles());
      select(null);
      setNotice('Deleted. Pictures already made with it keep the settings they were made with.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  const canSave = dirty && draft.name.trim() !== '';
  const count = profiles?.length ?? 0;

  return (
    <section className="settings-section">
      <h3 className="settings-section__title">Your models</h3>
      <p className="settings-hint">
        A model here is another file set for a family the app already supports - a fine-tune, say, or a different checkpoint. Pick it
        on the Generate page. Steps and CFG are the settings it was set up for; it is up to you to set what suits it.
      </p>
      <div className="styles-page__body">
        <aside className="panel styles-page__list">
          <div className="styles-page__list-head">
            <h4 className="panel__title">Models{profiles ? ` (${count + 1})` : ''}</h4>
            <button type="button" className="primary" onClick={() => select(null)} disabled={selectedId === null && !dirty}>
              + New
            </button>
          </div>
          <ul>
            <li>
              <div className="styles-page__item models-profile__builtin">
                <span className="styles-page__item-name">{def.builtInName}</span>
                <span className="styles-page__item-text">Built-in - the model KVGenius ships with</span>
              </div>
            </li>
            {profiles?.map((p) => (
              <li key={p.id}>
                <button
                  type="button"
                  className={`styles-page__item${p.id === selectedId ? ' styles-page__item--active' : ''}`}
                  onClick={() => select(p)}
                >
                  <span className="styles-page__item-name">{p.name}</span>
                  <span className="styles-page__item-text">
                    {p.sampler.steps} steps, CFG {p.sampler.cfg}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <div className="panel styles-page__editor">
          <div className="styles-page__editor-head">
            <h4 className="panel__title">{selected ? `Edit "${selected.name}"` : 'New model'}</h4>
            {dirty && <span className="styles-page__unsaved">Unsaved changes</span>}
          </div>

          <div className="styles-page__field">
            <div className="styles-page__field-head">
              <label className="field-label" htmlFor="model-name">
                Name
              </label>
              <span className="styles-page__count">
                {draft.name.length} / {MAX_PROFILE_NAME_LENGTH}
              </span>
            </div>
            <input
              id="model-name"
              type="text"
              value={draft.name}
              maxLength={MAX_PROFILE_NAME_LENGTH}
              onChange={(e) => setField('name', e.target.value)}
              placeholder="e.g. Photoreal v2"
              style={{ width: '100%' }}
            />
          </div>

          {PROFILE_FAMILIES.length > 1 ? (
            <div className="styles-page__field">
              <label className="field-label" htmlFor="model-family">
                Kind of model
              </label>
              <select id="model-family" value={draft.family} onChange={(e) => setDraft(newDraftKeepingName(draft, e.target.value))} style={{ width: '100%' }}>
                {PROFILE_FAMILIES.map((f) => (
                  <option key={f.family} value={f.family}>
                    {f.label}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <p className="settings-hint">Kind: {def.label}</p>
          )}

          <h4 className="models-profile__heading">Files</h4>
          {def.slots.map((slot) => {
            const options = installed ? installed[slot.folder] : null;
            const value = draft.files[slot.key] ?? '';
            const missing = options !== null && value !== '' && !options.includes(value);
            return (
              <div className="styles-page__field" key={slot.key}>
                <label className="field-label" htmlFor={`slot-${slot.key}`}>
                  {slot.label} <span className="models-profile__folder">({slot.folder})</span>
                </label>
                {options === null ? (
                  <input id={`slot-${slot.key}`} type="text" value={value} onChange={(e) => setFile(slot.key, e.target.value)} style={{ width: '100%' }} />
                ) : (
                  <select id={`slot-${slot.key}`} value={value} onChange={(e) => setFile(slot.key, e.target.value)} style={{ width: '100%' }}>
                    {missing && <option value={value}>{value} (not found in ComfyUI)</option>}
                    {options.map((file) => (
                      <option key={file} value={file}>
                        {file}
                      </option>
                    ))}
                    {options.length === 0 && !missing && <option value="">No files found in {slot.folder}</option>}
                  </select>
                )}
                {missing && <p className="models-profile__warn">ComfyUI does not list this file. Put it in {slot.folder} and use Check Again above.</p>}
              </div>
            );
          })}
          <FilesNote installed={installed} />

          <h4 className="models-profile__heading">Settings</h4>
          <div className="models-profile__grid">
            <label>
              <span className="field-label">Steps</span>
              <input type="number" min={SAMPLER_LIMITS.steps.min} max={SAMPLER_LIMITS.steps.max} value={draft.steps} onChange={(e) => setField('steps', e.target.value)} />
            </label>
            <label>
              <span className="field-label">CFG</span>
              <input type="number" step={0.1} min={SAMPLER_LIMITS.cfg.min} max={SAMPLER_LIMITS.cfg.max} value={draft.cfg} onChange={(e) => setField('cfg', e.target.value)} />
            </label>
            <label>
              <span className="field-label">Shift</span>
              <input type="number" step={0.5} min={SAMPLER_LIMITS.shift.min} max={SAMPLER_LIMITS.shift.max} value={draft.shift} onChange={(e) => setField('shift', e.target.value)} />
            </label>
            <label>
              <span className="field-label">Sampler</span>
              <ChoiceField value={draft.sampler} choices={choices.samplers} onChange={(v) => setField('sampler', v)} />
            </label>
            <label>
              <span className="field-label">Scheduler</span>
              <ChoiceField value={draft.scheduler} choices={choices.schedulers} onChange={(v) => setField('scheduler', v)} />
            </label>
          </div>
          {diffusionFile && diffusionFile !== builtInFile && (
            <p className="settings-hint">
              {DISTILLED_NAME.test(diffusionFile)
                ? 'The file name suggests a distilled model, which wants few steps and a CFG near 1 - like the starting values here.'
                : 'The starting values are the built-in Turbo model\'s, which is distilled. If this model is not, it probably wants more steps and a higher CFG - check its model page for what it recommends.'}
            </p>
          )}

          <div className="styles-page__actions">
            <button type="button" className="primary" onClick={() => void handleSave()} disabled={!canSave}>
              {selected ? 'Save changes' : 'Save model'}
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
        </div>
      </div>
    </section>
  );
}

function newDraftKeepingName(current: Draft, family: string): Draft {
  return { ...newDraft(family), name: current.name };
}

function FilesNote({ installed }: { installed: unknown }) {
  return installed === null ? (
    <p className="settings-hint">ComfyUI is not reachable and no models folder is set, so file names are typed by hand. Start ComfyUI to pick from its list.</p>
  ) : null;
}

/** A dropdown of the names ComfyUI offers, or a plain text box when they are not known. */
function ChoiceField({ value, choices, onChange }: { value: string; choices: string[]; onChange: (value: string) => void }) {
  if (choices.length === 0) return <input type="text" value={value} onChange={(e) => onChange(e.target.value)} />;
  const withCurrent = choices.includes(value) || value === '' ? choices : [value, ...choices];
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {withCurrent.map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
    </select>
  );
}
