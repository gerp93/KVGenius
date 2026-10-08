import { useEffect, useState } from 'react';
import { NO_SAMPLER, PROFILE_FAMILIES, SAMPLER_LIMITS, UPSCALE_IMPORT_FAMILY, UPSCALE_IMPORT_SLOT, profileFamily } from '../../shared/modelFamilies';
import { MAX_PROFILE_NAME_LENGTH, ModelProfile, SamplerSettings } from '../../shared/modelProfiles';
import { MODEL_MANIFEST, manifestFeature } from '../../shared/modelManifest';
import { ModelStatusReport, readinessLabel, summarize, summarizeSlots } from '../../shared/modelStatus';
import { FAMILY_KIND } from '../../shared/types';
import { ImageSettingsResult, ModelTestResult } from '../../shared/modelCheck';
import './Stepper.css';
import ModelFileImport from './ModelFileImport';
import ModelDownload from './ModelDownload';
import ModelFilesTable, { ChosenModelsTable, summaryText } from './ModelFilesTable';

interface Props {
  /** What ComfyUI (or the models folder) has, to choose files from. null while the first check runs. */
  report: ModelStatusReport | null;
  /** Tells the app a model was added, edited or deleted, so Generate's dropdown can reload. */
  onChanged: () => void;
  /** ComfyUI's models folder is known, so files can be copied into it. */
  canImport: boolean;
  /** A file was copied into ComfyUI's folder: look again at what it has. */
  onFilesChanged: () => void;
  /** ComfyUI's models folder when it is usable (where downloads go), else null. */
  modelsDir: string | null;
}

/** What the right-hand panel shows: a starter model (read-only), one of the user's own (or a new one), or the upscale models. */
type PanelMode = 'starter' | 'edit' | 'upscale';

/** What a new model is: the first choice of the new-model wizard. */
type NewType = 'image' | 'video' | 'upscale';
type WizardStep = 'type' | 'family' | 'files' | 'settings' | 'review' | 'upscale-file';

const NEW_TYPES: { type: NewType; icon: string; label: string; text: string }[] = [
  { type: 'image', icon: '🖼️', label: 'Image', text: 'Makes images from a prompt, or from another image.' },
  { type: 'video', icon: '🎬', label: 'Video', text: 'Makes a short clip from an image.' },
  { type: 'upscale', icon: '🔍', label: 'Upscaling', text: 'Enlarges images and videos. One file, nothing else to set.' },
];

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
  const s: SamplerSettings = def.sampler ?? NO_SAMPLER;
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
export default function ModelProfiles({ report, onChanged, canImport, onFilesChanged, modelsDir }: Props) {
  const [profiles, setProfiles] = useState<ModelProfile[] | null>(null);
  const [mode, setMode] = useState<PanelMode>('starter');
  // The new-model wizard: which step it is on, and the type chosen in its first step.
  const [newStep, setNewStep] = useState(0);
  const [newType, setNewType] = useState<NewType>('image');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [draft, setDraft] = useState<Draft>(newDraft());
  const [saved, setSaved] = useState<Draft>(newDraft());
  const [choices, setChoices] = useState<{ samplers: string[]; schedulers: string[] }>({ samplers: [], schedulers: [] });
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ModelTestResult | null>(null);
  const [imageNote, setImageNote] = useState<string | null>(null);

  // A test result describes the settings as they were; once anything changes it no longer applies.
  const draftKey = JSON.stringify(draft);
  useEffect(() => setTestResult(null), [draftKey]);

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

  function selectStarter(family: string) {
    const next = newDraft(family);
    setMode('starter');
    setSelectedId(null);
    setDraft(next);
    setSaved(next);
    setError(null);
    setNotice(null);
    setConfirmingDelete(false);
    setImageNote(null);
  }

  function selectUpscale() {
    setMode('upscale');
    setSelectedId(null);
    setError(null);
    setNotice(null);
  }

  function select(profile: ModelProfile | null) {
    const next = profile ? draftFrom(profile) : newDraft();
    setMode('edit');
    setNewStep(0);
    setNewType('image');
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

  function currentInput() {
    return {
      family: draft.family,
      name: draft.name,
      files: draft.files,
      sampler: { steps: Number(draft.steps), cfg: Number(draft.cfg), shift: Number(draft.shift), sampler: draft.sampler, scheduler: draft.scheduler },
    };
  }

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      setTestResult(await window.kvgenius.testModelProfile(currentInput()));
    } catch (err) {
      setTestResult({ ok: false, message: cleanError(err) });
    } finally {
      setTesting(false);
    }
  }

  /** Fills steps, CFG, sampler, scheduler and shift from what a picture says about how it was made. */
  async function handleReadImage() {
    setImageNote(null);
    let result: ImageSettingsResult | null;
    try {
      result = await window.kvgenius.readImageSettings();
    } catch (err) {
      setImageNote(cleanError(err));
      return;
    }
    if (!result) return;
    const s = result.settings;
    if (!s) {
      setImageNote(`${result.fileName} has no settings saved in it (many sites remove them). An image saved straight from ComfyUI or KVGenius has them.`);
      return;
    }
    const known = (value: string | undefined, list: string[]) => value !== undefined && (list.length === 0 || list.includes(value));
    const parts: string[] = [];
    setDraft((d) => {
      const next = { ...d };
      if (s.steps !== undefined) {
        next.steps = String(s.steps);
        parts.push(`${s.steps} steps`);
      }
      if (s.cfg !== undefined) {
        next.cfg = String(s.cfg);
        parts.push(`CFG ${s.cfg}`);
      }
      if (s.shift !== undefined) {
        next.shift = String(s.shift);
        parts.push(`shift ${s.shift}`);
      }
      if (s.sampler !== undefined && known(s.sampler, choices.samplers)) {
        next.sampler = s.sampler;
        parts.push(s.sampler);
      }
      if (s.scheduler !== undefined && known(s.scheduler, choices.schedulers)) {
        next.scheduler = s.scheduler;
        parts.push(s.scheduler);
      }
      return next;
    });
    const hint = s.fileHints?.diffusionModel;
    const where = s.source === 'comfyui' ? "the image's ComfyUI workflow" : 'its parameters text';
    setImageNote(
      `Read from ${result.fileName} (${where}): ${parts.join(', ') || 'nothing usable'}.${
        s.source === 'a1111' ? " Its sampler names differ from ComfyUI's, so those were left alone." : ''
      }${hint ? ` It used the image model ${hint}.` : ''}`,
    );
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
      setNotice('Deleted. Images already made with it keep the settings they were made with.');
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  const canSave = dirty && draft.name.trim() !== '';
  const count = profiles?.length ?? 0;
  const starter = mode === 'starter';
  const creating = mode === 'edit' && selectedId === null;
  const feature = manifestFeature(def.family) ?? MODEL_MANIFEST.find((f) => f.family === def.family);
  const upscaleFeature = manifestFeature('upscale');
  const upscaleModels = report && report.source !== 'none' ? report.installed.upscale_models : null;
  const totalModels = PROFILE_FAMILIES.length + count + 1;

  /** One list entry's subtitle and readiness, for a starter model (its manifest files) or a saved one (its slots). */
  function readiness(family: string, files: Record<string, string> | null) {
    if (!report || report.source === 'none') return null;
    const fam = profileFamily(family);
    if (!fam) return null;
    if (files) return readinessLabel(summarizeSlots(fam.slots, files, report));
    const feat = MODEL_MANIFEST.find((f) => f.family === family);
    return readinessLabel(feat ? summarize(feat.files, report) : null);
  }

  const KINDS: { kind: 'image' | 'video'; label: string }[] = [
    { kind: 'image', label: 'Image' },
    { kind: 'video', label: 'Video' },
  ];
  const sub = (family: string, sampler: SamplerSettings | null) =>
    profileFamily(family)?.sampler && sampler ? `${sampler.steps} steps, CFG ${sampler.cfg}` : profileFamily(family)?.sampler ? '' : 'Fast / High';

  // ---- the new-model wizard: type -> (family, when the type has several) -> name and files -> settings -> test and save ----
  const kindFamilies = PROFILE_FAMILIES.filter((f) => FAMILY_KIND[f.family] === newType);
  const stepIds: WizardStep[] =
    newType === 'upscale'
      ? ['type', 'upscale-file']
      : ['type', ...(kindFamilies.length > 1 ? (['family'] as WizardStep[]) : []), 'files', ...(def.sampler ? (['settings'] as WizardStep[]) : []), 'review'];
  const stepIndex = Math.min(newStep, stepIds.length - 1);
  const stepId = stepIds[stepIndex];
  const filesComplete = draft.name.trim() !== '' && def.slots.every((slot) => (draft.files[slot.key] ?? '').trim() !== '');

  function pickType(type: NewType) {
    setNewType(type);
    if (type !== 'upscale') {
      const first = PROFILE_FAMILIES.find((f) => FAMILY_KIND[f.family] === type);
      if (first && first.family !== draft.family) setDraft(newDraftKeepingName(draft, first.family));
    }
  }

  const nameField = (
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
  );

  /** Where each loader slot's file is chosen (a list of what ComfyUI has, or typed when that is not known), with the import control. */
  const filesEditor = (
    <>
      {report && report.source !== 'none' && (
        <div className="models-feature__summary">
          {(() => {
            const s = summarizeSlots(def.slots, draft.files, report);
            return s.present === s.total ? `All ${s.total} installed` : `${s.present} of ${s.total} installed`;
          })()}
        </div>
      )}
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
            <ModelFileImport
              family={draft.family}
              slot={slot}
              canImport={canImport}
              onImported={(fileName) => {
                setFile(slot.key, fileName);
                onFilesChanged();
              }}
            />
          </div>
        );
      })}
      <FilesNote installed={installed} />
    </>
  );

  const settingsBlock = def.sampler ? (
    <>
      {!starter && (
        <div className="button-row" style={{ marginBottom: 8 }}>
          <button type="button" onClick={() => void handleReadImage()}>
            Read settings from an image...
          </button>
        </div>
      )}
      {imageNote && <p className="settings-hint">{imageNote}</p>}
      <div className="models-profile__grid">
        <label>
          <span className="field-label">Steps</span>
          <input type="number" disabled={starter} min={SAMPLER_LIMITS.steps.min} max={SAMPLER_LIMITS.steps.max} value={draft.steps} onChange={(e) => setField('steps', e.target.value)} />
        </label>
        <label>
          <span className="field-label">CFG</span>
          <input type="number" disabled={starter} step={0.1} min={SAMPLER_LIMITS.cfg.min} max={SAMPLER_LIMITS.cfg.max} value={draft.cfg} onChange={(e) => setField('cfg', e.target.value)} />
        </label>
        <label>
          <span className="field-label">Shift</span>
          <input type="number" disabled={starter} step={0.5} min={SAMPLER_LIMITS.shift.min} max={SAMPLER_LIMITS.shift.max} value={draft.shift} onChange={(e) => setField('shift', e.target.value)} />
        </label>
        <label>
          <span className="field-label">Sampler</span>
          <ChoiceField disabled={starter} value={draft.sampler} choices={choices.samplers} onChange={(v) => setField('sampler', v)} />
        </label>
        <label>
          <span className="field-label">Scheduler</span>
          <ChoiceField disabled={starter} value={draft.scheduler} choices={choices.schedulers} onChange={(v) => setField('scheduler', v)} />
        </label>
      </div>
      {!starter && diffusionFile && diffusionFile !== builtInFile && (
        <p className="settings-hint">
          {DISTILLED_NAME.test(diffusionFile)
            ? 'The file name suggests a distilled model, which wants few steps and a CFG near 1 - like the starting values here.'
            : `The starting values are ${def.builtInName}'s, which is distilled. If this model is not, it probably wants more steps and a higher CFG - check its model page for what it recommends.`}
        </p>
      )}
    </>
  ) : (
    <p className="settings-hint">A video model only changes which files are used. Quality (Fast or High) is chosen on the Generate page.</p>
  );

  const testBlock = (
    <>
      <div className="button-row" style={{ marginTop: 12 }}>
        <button type="button" onClick={() => void handleTest()} disabled={testing}>
          {testing ? 'Testing...' : 'Test this model'}
        </button>
        <span className="settings-hint" style={{ margin: 0 }}>
          {def.sampler
            ? 'One small image, to see that the files load and run. Needs ComfyUI running and the queue empty.'
            : 'One tiny clip, to see that the files load and run. The video models are large, so this can take a few minutes. Needs ComfyUI running and the queue empty.'}
        </span>
      </div>
      {testResult && (
        <div className={`model-test model-test--${testResult.ok ? 'ok' : 'failed'}`}>
          {testResult.ok ? '✓ ' : '✗ '}
          {testResult.message}
          {testResult.imageBase64 && testResult.mime?.startsWith('video/') ? (
            <video src={`data:${testResult.mime};base64,${testResult.imageBase64}`} autoPlay loop muted controls />
          ) : (
            testResult.imageBase64 && <img src={`data:${testResult.mime ?? 'image/png'};base64,${testResult.imageBase64}`} alt="The test image" />
          )}
        </div>
      )}
    </>
  );

  const saveMessages = (
    <>
      {error && <span className="styles-page__error">{error}</span>}
      {!error && notice && <span className="styles-page__notice">{notice}</span>}
    </>
  );

  const stepTitle: Record<WizardStep, string> = {
    type: 'What kind of model?',
    family: 'Which family?',
    files: 'Name and files',
    settings: 'Settings',
    review: 'Test and save',
    'upscale-file': 'Choose the file',
  };

  const wizardBody = (
    <>
      {stepId === 'type' && (
        <div className="model-wizard__choices" role="radiogroup" aria-label="Kind of model">
          {NEW_TYPES.map((t) => (
            <button
              key={t.type}
              type="button"
              role="radio"
              aria-checked={newType === t.type}
              className={`model-wizard__choice${newType === t.type ? ' model-wizard__choice--on' : ''}`}
              onClick={() => pickType(t.type)}
            >
              <span className="model-wizard__choice-name">
                {t.icon} {t.label}
              </span>
              <span className="model-wizard__choice-text">{t.text}</span>
            </button>
          ))}
        </div>
      )}
      {stepId === 'family' && (
        <div className="model-wizard__choices" role="radiogroup" aria-label="Family">
          {kindFamilies.map((f) => (
            <button
              key={f.family}
              type="button"
              role="radio"
              aria-checked={draft.family === f.family}
              className={`model-wizard__choice${draft.family === f.family ? ' model-wizard__choice--on' : ''}`}
              onClick={() => setDraft(newDraftKeepingName(draft, f.family))}
            >
              <span className="model-wizard__choice-name">{f.label}</span>
              <span className="model-wizard__choice-text">Starts from {f.builtInName}&apos;s files{f.sampler ? ' and settings' : ''}.</span>
            </button>
          ))}
        </div>
      )}
      {stepId === 'files' && (
        <>
          {nameField}
          <h4 className="models-profile__heading">Files</h4>
          {filesEditor}
        </>
      )}
      {stepId === 'settings' && settingsBlock}
      {stepId === 'review' && (
        <>
          <p className="settings-hint">
            <strong>{draft.name.trim() || 'Unnamed'}</strong> - {def.label}
            {report && report.source !== 'none' && (() => {
              const s = summarizeSlots(def.slots, draft.files, report);
              return `, ${s.present} of ${s.total} files installed`;
            })()}
            .
          </p>
          {testBlock}
        </>
      )}
      {stepId === 'upscale-file' && (
        <>
          <p className="settings-hint">
            An upscale model is a single file, picked each time on the Upscale page - nothing else to set. Choose it here and it is copied
            into ComfyUI&apos;s upscale_models folder.
          </p>
          <ModelFileImport family={UPSCALE_IMPORT_FAMILY} slot={UPSCALE_IMPORT_SLOT} canImport={canImport} onImported={() => onFilesChanged()} />
        </>
      )}
    </>
  );

  const lastStep = stepIndex === stepIds.length - 1;
  const canNext = stepId === 'files' ? filesComplete : true;

  return (
    <section className="settings-section">
      <h3 className="settings-section__title">Models</h3>
      <p className="settings-hint">
        Every model KVGenius can use, with the files each one needs. Add your own for a kind the app already supports - a fine-tune, say, or
        a different checkpoint - and pick it on the Generate page. Steps and CFG are the settings a model was set up for; it is up to you
        to set what suits it.
      </p>
      <div className="styles-page__body">
        <aside className="panel styles-page__list">
          <div className="styles-page__list-head">
            <h4 className="panel__title">Models{profiles ? ` (${totalModels - 1})` : ''}</h4>
            <button type="button" className="primary" onClick={() => select(null)} disabled={creating && newStep === 0 && !dirty}>
              + New
            </button>
          </div>
          {KINDS.map(({ kind, label }) => (
            <div key={kind}>
              <div className="models-list__group">{label}</div>
              <ul>
                {PROFILE_FAMILIES.filter((f) => FAMILY_KIND[f.family] === kind).map((f) => (
                  <Entry
                    key={f.family}
                    active={starter && draft.family === f.family}
                    onClick={() => selectStarter(f.family)}
                    name={f.builtInName}
                    text={sub(f.family, f.sampler)}
                    status={readiness(f.family, null)}
                  />
                ))}
                {profiles
                  ?.filter((p) => FAMILY_KIND[p.family] === kind)
                  .map((p) => (
                    <Entry
                      key={p.id}
                      active={mode === 'edit' && p.id === selectedId}
                      onClick={() => select(p)}
                      name={p.name}
                      text={sub(p.family, p.sampler)}
                      status={readiness(p.family, p.files)}
                    />
                  ))}
              </ul>
            </div>
          ))}
          <div>
            <div className="models-list__group">Upscaling</div>
            <ul>
              <Entry
                active={mode === 'upscale'}
                onClick={selectUpscale}
                name="Upscale models"
                text="you choose"
                status={upscaleModels ? { ok: upscaleModels.length > 0, text: upscaleModels.length > 0 ? `${upscaleModels.length} installed` : 'none installed' } : null}
              />
            </ul>
          </div>
        </aside>

        <div className="panel styles-page__editor">
          {mode === 'upscale' ? (
            <>
              <div className="styles-page__editor-head">
                <h4 className="panel__title">{upscaleFeature?.title ?? 'Upscaling'}</h4>
              </div>
              <p className="settings-hint">{upscaleFeature?.summary}</p>
              {upscaleModels ? (
                <ChosenModelsTable
                  folder="upscale_models"
                  role="Upscale model"
                  names={upscaleModels}
                  emptyText="No upscale models installed yet. Use + New to add one."
                />
              ) : (
                <p className="settings-hint">Can&apos;t tell which upscale models are installed until ComfyUI is reachable or its models folder is set.</p>
              )}
            </>
          ) : creating ? (
            <>
              <div className="styles-page__editor-head">
                <h4 className="panel__title">New model</h4>
                {dirty && <span className="styles-page__unsaved">Unsaved changes</span>}
              </div>
              <div className="stepper__progress" aria-live="polite">
                <span className="stepper__count">
                  Step {stepIndex + 1} of {stepIds.length} - {stepTitle[stepId]}
                </span>
                <div className="stepper__track" aria-hidden>
                  <div className="stepper__fill" style={{ width: `${((stepIndex + 1) / stepIds.length) * 100}%` }} />
                </div>
              </div>
              <h4 className="models-profile__heading">{stepTitle[stepId]}</h4>
              {wizardBody}
              <div className="styles-page__actions">
                {stepIndex > 0 && (
                  <button type="button" onClick={() => setNewStep(stepIndex - 1)}>
                    Back
                  </button>
                )}
                {stepId === 'upscale-file' ? (
                  <button type="button" className="primary" onClick={selectUpscale}>
                    Done
                  </button>
                ) : lastStep ? (
                  <button type="button" className="primary" onClick={() => void handleSave()} disabled={!canSave}>
                    Save model
                  </button>
                ) : (
                  <button type="button" className="primary" onClick={() => setNewStep(stepIndex + 1)} disabled={!canNext}>
                    Next
                  </button>
                )}
                {stepId === 'files' && !filesComplete && <span className="settings-hint" style={{ margin: 0 }}>Give it a name and a file for each slot.</span>}
                {saveMessages}
              </div>
            </>
          ) : (
            <>
              <div className="styles-page__editor-head">
                <h4 className="panel__title">{starter ? def.builtInName : `Edit "${selected?.name ?? ''}"`}</h4>
                {dirty && !starter && <span className="styles-page__unsaved">Unsaved changes</span>}
              </div>
              {starter && feature && <p className="settings-hint">{feature.summary}</p>}
              {!starter && (
                <>
                  {nameField}
                  <p className="settings-hint">Family: {def.label}</p>
                </>
              )}

              <h4 className="models-profile__heading">Files</h4>
              {starter && feature ? (
                <>
                  <ModelFilesTable feature={feature} report={report} modelsDir={modelsDir} />
                  {summaryText(feature, report) && <div className="models-feature__summary">{summaryText(feature, report)}</div>}
                  <ModelDownload feature={feature} report={report} modelsDir={modelsDir} />
                  {feature.note && <p className="settings-hint">{feature.note}</p>}
                </>
              ) : (
                filesEditor
              )}

              {def.sampler && <h4 className="models-profile__heading">Settings</h4>}
              {settingsBlock}
              {testBlock}

              {!starter && (
                <div className="styles-page__actions">
                  <button type="button" className="primary" onClick={() => void handleSave()} disabled={!canSave}>
                    Save changes
                  </button>
                  {saveMessages}
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
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

/** One row of the models list: name, what it is set up for, and whether its files are all there. */
function Entry({ active, onClick, name, text, status }: { active: boolean; onClick: () => void; name: string; text: string; status: { ok: boolean; text: string } | null }) {
  return (
    <li>
      <button type="button" className={`styles-page__item${active ? ' styles-page__item--active' : ''}`} onClick={onClick}>
        <span className="styles-page__item-name">{name}</span>
        <span className="styles-page__item-text models-list__sub">
          <span>{text}</span>
          {status && <span className={status.ok ? 'models-list__ok' : 'models-list__bad'}>{status.ok ? '✓' : '✗'} {status.text}</span>}
        </span>
      </button>
    </li>
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
function ChoiceField({ value, choices, onChange, disabled }: { value: string; choices: string[]; onChange: (value: string) => void; disabled?: boolean }) {
  if (choices.length === 0) return <input type="text" disabled={disabled} value={value} onChange={(e) => onChange(e.target.value)} />;
  const withCurrent = choices.includes(value) || value === '' ? choices : [value, ...choices];
  return (
    <select disabled={disabled} value={value} onChange={(e) => onChange(e.target.value)}>
      {withCurrent.map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
    </select>
  );
}
