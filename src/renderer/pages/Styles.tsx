import { useCallback, useEffect, useMemo, useState } from "react";
import {
  MAX_STYLE_NAME_LENGTH,
  MAX_STYLE_TEXT_LENGTH,
  PromptStyle,
  STYLE_KINDS,
  STYLE_KIND_LABEL,
  StyleKind,
  combinePrompt,
} from "../../shared/styles";
import {
  BASELINE_ID,
  DEFAULT_SAMPLE_PROMPT,
  DEFAULT_SAMPLE_SEED,
  MAX_SAMPLE_PROMPT_LENGTH,
  SampleState,
  StyleSampleInfo,
  StyleSamplesView,
  idsToRender,
} from "../../shared/styleSamples";

interface Props {
  /** Tells the app a style was added, edited or deleted, so Generate's dropdown can reload. */
  onChanged: () => void;
}

/** Electron prefixes errors thrown in an ipcMain handler with "Error invoking remote method". */
function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.replace(
    /^Error invoking remote method '[^']+': (Error: )?/,
    "",
  );
}

function formatDay(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
}

const STATE_LABEL: Partial<Record<SampleState, string>> = {
  outdated: "Outdated",
  queued: "Queued",
  rendering: "Rendering...",
  failed: "Failed",
};

/** The example picture of a card or of the editor: the picture (marked while it is being redone or out of date), or why there is none. */
function SamplePicture({
  sample,
  large = false,
}: {
  sample: StyleSampleInfo | undefined;
  large?: boolean;
}) {
  const label = sample ? STATE_LABEL[sample.state] : undefined;
  return (
    <div
      className={`style-sample${large ? " style-sample--large" : ""}${sample?.state === "outdated" ? " style-sample--outdated" : ""}`}
    >
      {sample?.imageUrl ? (
        <img
          src={sample.imageUrl}
          alt=""
          loading="lazy"
          decoding="async"
          draggable={false}
        />
      ) : (
        <span className="style-sample__none">
          {sample?.state === "queued" || sample?.state === "rendering"
            ? ""
            : "No example yet"}
        </span>
      )}
      {label && (
        <span
          className={`style-sample__badge style-sample__badge--${sample?.state}`}
        >
          {label}
        </span>
      )}
    </div>
  );
}

/**
 * Where the user defines their reusable prompt wording, of two kinds: a Style (a general look - "1930s
 * movie poster, bold lithograph"; one per picture) and an Element (a part of the picture used again and
 * again - an outfit, a character; any number). Every one has an example picture, all made from the same
 * standard prompt and seed (and one with nothing added, the baseline), so what each does is easy to compare.
 * About three quarters of the page is the examples; the editor is a side panel.
 */
export default function Styles({ onChanged }: Props) {
  const [styles, setStyles] = useState<PromptStyle[] | null>(null);
  const [samples, setSamples] = useState<StyleSamplesView | null>(null);
  // The style being edited, or null while writing a new one.
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [kind, setKind] = useState<StyleKind>("style");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  // The standard prompt and seed being typed; saved with the button (or when rendering).
  const [promptDraft, setPromptDraft] = useState<string | null>(null);
  const [seedDraft, setSeedDraft] = useState("");
  const [sampleError, setSampleError] = useState<string | null>(null);

  const loadSamples = useCallback(async () => {
    try {
      setSamples(await window.kvgenius.getStyleSamples());
    } catch (err) {
      setSampleError(cleanError(err));
    }
  }, []);

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
    void loadSamples();
    return () => {
      cancelled = true;
    };
  }, [loadSamples]);

  // Take the saved standard prompt and seed into the form the first time they arrive.
  useEffect(() => {
    if (samples && promptDraft === null) {
      setPromptDraft(samples.settings.prompt);
      setSeedDraft(String(samples.settings.seed));
    }
  }, [samples, promptDraft]);

  const working =
    samples?.samples.some(
      (s) => s.state === "queued" || s.state === "rendering",
    ) ?? false;
  // While examples are being made, look again every couple of seconds to pick them up as they finish.
  useEffect(() => {
    if (!working) return;
    const timer = window.setInterval(() => void loadSamples(), 2000);
    return () => window.clearInterval(timer);
  }, [working, loadSamples]);

  const sampleOf = useMemo(
    () => new Map((samples?.samples ?? []).map((s) => [s.id, s])),
    [samples],
  );
  const selected = styles?.find((s) => s.id === selectedId) ?? null;
  const dirty = selected
    ? name !== selected.name || text !== selected.text || kind !== selected.kind
    : name.trim() !== "" || text.trim() !== "";
  const settingsDirty =
    samples !== null &&
    promptDraft !== null &&
    (promptDraft.trim() !== samples.settings.prompt ||
      seedDraft.trim() !== String(samples.settings.seed));

  function select(style: PromptStyle | null) {
    setSelectedId(style ? style.id : null);
    setName(style?.name ?? "");
    setText(style?.text ?? "");
    setKind(style?.kind ?? "style");
    setError(null);
    setNotice(null);
    setConfirmingDelete(false);
  }

  async function handleSave() {
    setError(null);
    setNotice(null);
    try {
      const saved = await window.kvgenius.saveStyle(
        { name, text, kind },
        selectedId,
      );
      setStyles(await window.kvgenius.listStyles());
      await loadSamples();
      setSelectedId(saved.id);
      setName(saved.name);
      setText(saved.text);
      setKind(saved.kind);
      setNotice("Saved.");
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
      await loadSamples();
      select(null);
      setNotice("Deleted. Pictures already made with it keep their prompt.");
      onChanged();
    } catch (err) {
      setError(cleanError(err));
    }
  }

  /** Saves the standard prompt and seed if they were edited. Returns false (with the reason shown) when they cannot be saved. */
  async function saveSettings(): Promise<boolean> {
    if (!settingsDirty || promptDraft === null) return true;
    setSampleError(null);
    try {
      const saved = await window.kvgenius.saveStyleSampleSettings({
        prompt: promptDraft,
        seed: Number(seedDraft),
      });
      setPromptDraft(saved.prompt);
      setSeedDraft(String(saved.seed));
      await loadSamples();
      return true;
    } catch (err) {
      setSampleError(cleanError(err));
      return false;
    }
  }

  async function render(ids: number[]) {
    if (ids.length === 0 || !(await saveSettings())) return;
    setSampleError(null);
    try {
      await window.kvgenius.renderStyleSamples(ids);
      await loadSamples();
    } catch (err) {
      setSampleError(cleanError(err));
    }
  }

  // What "render" would make: with unsaved settings every example is out of date, so count against the settings as typed.
  const all = samples?.samples ?? [];
  const toRender = settingsDirty
    ? idsToRender(all, true)
    : idsToRender(all, false);
  const canSave = dirty && name.trim() !== "" && text.trim() !== "";
  const label = STYLE_KIND_LABEL[kind].toLowerCase();
  const samplePrompt =
    (promptDraft ?? samples?.settings.prompt ?? DEFAULT_SAMPLE_PROMPT).trim() ||
    DEFAULT_SAMPLE_PROMPT;
  const failure = all.find((s) => s.state === "failed" && s.error);

  function card(style: PromptStyle) {
    const sample = sampleOf.get(style.id);
    const outdated =
      sample?.state === "outdated" ||
      sample?.state === "none" ||
      sample?.state === "failed";
    return (
      <div
        key={style.id}
        className={`style-card${style.id === selectedId ? " style-card--active" : ""}`}
      >
        <button
          type="button"
          className="style-card__main"
          onClick={() => select(style)}
        >
          <SamplePicture sample={sample} />
          <span className="style-card__name">{style.name}</span>
          <span className="style-card__text">{style.text}</span>
        </button>
        <div className="style-card__foot">
          <span
            className="style-card__date"
            title="When this wording was last written or changed"
          >
            Changed {formatDay(style.textChangedAt)}
          </span>
          <button
            type="button"
            className={outdated ? "primary" : undefined}
            onClick={() => void render([style.id])}
            disabled={
              sample?.state === "queued" || sample?.state === "rendering"
            }
            title={
              outdated
                ? "Make this example again from the wording as it is now"
                : "Make this example again"
            }
          >
            {sample?.state === "none" ? "Render" : "Re-render"}
          </button>
        </div>
      </div>
    );
  }

  const baseline = sampleOf.get(BASELINE_ID);

  const baselineCard = (
    <div className="style-card style-card--baseline">
      <div className="style-card__main style-card__main--static">
        <SamplePicture sample={baseline} />
        <span className="style-card__name">No style</span>
        <span className="style-card__text">
          The standard prompt on its own - what every example is compared with.
        </span>
      </div>
      <div className="style-card__foot">
        <span className="style-card__date">Baseline</span>
        <button
          type="button"
          className={
            baseline && baseline.state !== "current" ? "primary" : undefined
          }
          onClick={() => void render([BASELINE_ID])}
          disabled={
            baseline?.state === "queued" || baseline?.state === "rendering"
          }
        >
          {baseline?.state === "none" || !baseline ? "Render" : "Re-render"}
        </button>
      </div>
    </div>
  );

  return (
    <div className="page">
      <div className="styles-page">
        <h2 className="styles-page__title">Styles</h2>
        <p className="styles-page__intro">
          A <strong>style</strong> is a general look - a picture uses one. An{" "}
          <strong>element</strong> is a part of the picture you use again and
          again - an outfit, a character - and a picture can use any number.
          Pick them on the Generate page: they are added after your prompt,
          elements first and the style last. With none picked, your prompt is
          sent exactly as you wrote it.
        </p>

        <div className="styles-page__body">
          <div className="styles-page__main">
            <section className="panel styles-page__standard">
              <div className="styles-page__editor-head">
                <h3 className="panel__title">
                  Standard prompt for the examples
                </h3>
                {settingsDirty && (
                  <span className="styles-page__unsaved">Unsaved changes</span>
                )}
              </div>
              <p className="styles-page__hint">
                Every example is made from this prompt with that style's or
                element's wording added, at the same seed - so the only thing
                that differs between them is the wording.
              </p>
              <div className="styles-page__standard-row">
                <textarea
                  value={promptDraft ?? ""}
                  maxLength={MAX_SAMPLE_PROMPT_LENGTH}
                  rows={2}
                  onChange={(e) => setPromptDraft(e.target.value)}
                  aria-label="Standard prompt"
                  placeholder={DEFAULT_SAMPLE_PROMPT}
                />
                <div className="styles-page__seed">
                  <label className="field-label" htmlFor="sample-seed">
                    Seed
                  </label>
                  <input
                    id="sample-seed"
                    type="number"
                    min={0}
                    value={seedDraft}
                    onChange={(e) => setSeedDraft(e.target.value)}
                  />
                </div>
              </div>
              <div className="styles-page__standard-actions">
                <button
                  type="button"
                  className="primary"
                  onClick={() => void render(toRender)}
                  disabled={samples === null || toRender.length === 0}
                >
                  {toRender.length === 0
                    ? "All examples are up to date"
                    : `Render ${toRender.length} example${toRender.length === 1 ? "" : "s"}`}
                </button>
                <button
                  type="button"
                  onClick={() => void render(idsToRender(all, true))}
                  disabled={samples === null || all.length === 0}
                >
                  Re-render all
                </button>
                <button
                  type="button"
                  onClick={() => void saveSettings()}
                  disabled={!settingsDirty}
                >
                  Save prompt
                </button>
                <button
                  type="button"
                  className="styles-page__link"
                  onClick={() => {
                    setPromptDraft(DEFAULT_SAMPLE_PROMPT);
                    setSeedDraft(String(DEFAULT_SAMPLE_SEED));
                  }}
                  title={`The default: "${DEFAULT_SAMPLE_PROMPT}", seed ${DEFAULT_SAMPLE_SEED}`}
                >
                  Use the default
                </button>
                {working && (
                  <span className="styles-page__notice">
                    Making examples - they appear here as they finish.
                  </span>
                )}
              </div>
              {sampleError && (
                <p className="styles-page__error">{sampleError}</p>
              )}
              {!sampleError && failure && (
                <p className="styles-page__error">
                  The last example failed: {failure.error}
                </p>
              )}
            </section>

            <section className="styles-page__group">
              <div className="styles-page__group-head">
                <h3 className="panel__title">
                  Your styles and elements{styles ? ` (${styles.length})` : ""}
                </h3>
                <button
                  type="button"
                  className="primary"
                  onClick={() => select(null)}
                  disabled={selectedId === null && !dirty}
                >
                  + New
                </button>
              </div>
              {styles === null ? (
                <p className="styles-page__empty">Loading...</p>
              ) : styles.length === 0 ? (
                <p className="styles-page__empty">
                  Nothing yet. Write your first one in the panel on the right.
                </p>
              ) : (
                STYLE_KINDS.map((k) => {
                  const items = styles.filter((s) => s.kind === k);
                  if (items.length === 0 && k !== "style") return null;
                  return (
                    <div key={k}>
                      <h4 className="styles-page__section">
                        {STYLE_KIND_LABEL[k]}s ({items.length})
                      </h4>
                      <div className="style-grid">
                        {k === "style" && baselineCard}
                        {items.map(card)}
                      </div>
                    </div>
                  );
                })
              )}
            </section>
          </div>

          <aside className="panel styles-page__editor">
            <div className="styles-page__editor-head">
              <h3 className="panel__title">
                {selected ? `Edit "${selected.name}"` : "New style or element"}
              </h3>
              {dirty && (
                <span className="styles-page__unsaved">Unsaved changes</span>
              )}
            </div>

            {selected && (
              <SamplePicture sample={sampleOf.get(selected.id)} large />
            )}

            <div className="styles-page__field">
              <span className="field-label">Kind</span>
              <div
                className="styles-page__kinds"
                role="radiogroup"
                aria-label="Kind"
              >
                {STYLE_KINDS.map((k) => (
                  <label key={k} className="styles-page__kind-option">
                    <input
                      type="radio"
                      name="style-kind"
                      checked={kind === k}
                      onChange={() => setKind(k)}
                    />{" "}
                    {STYLE_KIND_LABEL[k]}
                    <span className="styles-page__kind-hint">
                      {k === "style"
                        ? " - a general look, one per picture"
                        : " - an outfit, a character; any number"}
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
                placeholder={
                  kind === "style"
                    ? "e.g. 1930s movie poster"
                    : "e.g. Red trench coat outfit"
                }
                style={{ width: "100%" }}
              />
            </div>

            <div className="styles-page__field">
              <div className="styles-page__field-head">
                <label className="field-label" htmlFor="style-text">
                  {kind === "style" ? "Style text" : "Element text"}
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
                placeholder={
                  kind === "style"
                    ? "The words that produce the look, added after your prompt"
                    : "The words that describe it, added after your prompt"
                }
                rows={6}
                style={{ width: "100%", resize: "vertical" }}
              />
              {selected && (
                <span className="styles-page__count">
                  Wording last changed {formatDay(selected.textChangedAt)}
                </span>
              )}
            </div>

            <div className="styles-page__preview">
              <span className="field-label">What the example sends</span>
              <code>
                {combinePrompt(
                  samplePrompt,
                  text.trim() || `your ${label} text`,
                )}
              </code>
            </div>

            <div className="styles-page__actions">
              <button
                type="button"
                className="primary"
                onClick={() => void handleSave()}
                disabled={!canSave}
              >
                {selected ? "Save changes" : `Save ${label}`}
              </button>
              {selected && (
                <button
                  type="button"
                  className={`button-danger${confirmingDelete ? " button-danger--armed" : ""}`}
                  onClick={() => void handleDelete()}
                  onBlur={() => setConfirmingDelete(false)}
                >
                  {confirmingDelete ? "Click again to delete" : "Delete"}
                </button>
              )}
            </div>
            {error && <span className="styles-page__error">{error}</span>}
            {!error && notice && (
              <span className="styles-page__notice">{notice}</span>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
