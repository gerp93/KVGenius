import { useCallback, useEffect, useRef, useState } from 'react';
import CopyButton from '../components/CopyButton';
import QueuePanel from '../components/QueuePanel';
import ResultViewer from '../components/ResultViewer';
import ExpandButton from '../components/Lightbox';
import { GenerationQueue, MAX_BATCH_SIZE, MAX_PENDING_JOBS } from '../hooks/useGenerationQueue';
import { usePromptSlots } from '../hooks/usePromptSlots';
import { MAX_PROMPT_SLOTS } from '../../shared/promptSlots';
import { formatDuration, formatElapsed, formatEstimate } from '../utils/format';
import { VIDEO_FPS, framesToSeconds, secondsToFrames } from '../utils/video';
import { FAMILY_KIND, GenerationRecord, TimeEstimate, VideoSourceRequest } from '../../shared/types';
import { VIDEO_QUALITY_SETTINGS, VideoQuality, videoQualityFromCfg } from '../../shared/videoQuality';

type Mode = 'image' | 'video';

const FAMILY_FOR_MODE: Record<Mode, string> = {
  image: 'z-image-turbo',
  video: 'wan22-i2v',
};

interface Props {
  queue: GenerationQueue;
  recallRecord: GenerationRecord | null;
  onRecalled: () => void;
  recallPrompt: string | null;
  onPromptRecalled: () => void;
  videoSource: VideoSourceRequest | null;
  onVideoSourceHandled: () => void;
}

// Long side of a video generated from an existing image (matches the 640px default).
const VIDEO_LONG_SIDE = 640;

function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 32);
}

/** `count` different random seeds - a batch of the same prompt is pointless with repeated ones. */
function uniqueRandomSeeds(count: number): number[] {
  const seeds = new Set<number>();
  while (seeds.size < count) seeds.add(randomSeed());
  return [...seeds];
}

interface SizePreset {
  label: string;
  width: number;
  height: number;
}

// Image sizes are multiples of 32/64; video sizes multiples of 16, which is what Wan needs.
const IMAGE_SIZE_PRESETS: SizePreset[] = [
  { label: 'Square (1:1)', width: 1024, height: 1024 },
  { label: 'Square, small (1:1)', width: 768, height: 768 },
  { label: 'Square, large (1:1)', width: 1280, height: 1280 },
  { label: 'Square, extra large (1:1)', width: 2048, height: 2048 },
  { label: 'Portrait (3:4)', width: 896, height: 1152 },
  { label: 'Portrait (2:3)', width: 832, height: 1216 },
  { label: 'Portrait (4:5)', width: 896, height: 1120 },
  { label: 'Portrait (9:16)', width: 768, height: 1344 },
  { label: 'Tall (9:21)', width: 640, height: 1536 },
  { label: 'Landscape (4:3)', width: 1152, height: 896 },
  { label: 'Landscape (3:2)', width: 1216, height: 832 },
  { label: 'Landscape (5:4)', width: 1120, height: 896 },
  { label: 'Landscape (16:9)', width: 1344, height: 768 },
  { label: 'Ultrawide (21:9)', width: 1536, height: 640 },
];

const VIDEO_SIZE_PRESETS: SizePreset[] = [
  { label: 'Square (1:1)', width: 640, height: 640 },
  { label: 'Square, small (1:1)', width: 480, height: 480 },
  { label: 'Portrait (3:4)', width: 480, height: 640 },
  { label: 'Portrait (2:3)', width: 432, height: 640 },
  { label: 'Portrait (9:16)', width: 368, height: 640 },
  { label: '480p portrait (9:16)', width: 480, height: 832 },
  { label: 'Landscape (4:3)', width: 640, height: 480 },
  { label: 'Landscape (3:2)', width: 640, height: 432 },
  { label: 'Landscape (16:9)', width: 640, height: 368 },
  { label: '480p landscape (16:9)', width: 832, height: 480 },
];

export default function Generate({
  queue,
  recallRecord,
  onRecalled,
  recallPrompt,
  onPromptRecalled,
  videoSource,
  onVideoSourceHandled,
}: Props) {
  // The left-hand form is one of several independent "tabs" (prompt + every setting below it),
  // switchable and persisted across restarts - see usePromptSlots for the field definitions.
  const slotState = usePromptSlots();
  const {
    slots,
    activeSlotId,
    switchTo: switchSlot,
    addSlot,
    closeSlot,
    renameSlot,
    labelFor: slotLabelFor,
    mode,
    setMode,
    prompt,
    setPrompt,
    width,
    setWidth,
    height,
    setHeight,
    seed,
    setSeed,
    seedLocked,
    setSeedLocked,
    steps,
    setSteps,
    cfg,
    setCfg,
    lengthSeconds,
    setLengthSeconds,
    videoQuality,
    setVideoQuality,
    sourceImagePath,
    setSourceImagePath,
    advancedOpen,
    setAdvancedOpen,
    customSize,
    setCustomSize,
    batchSize,
    setBatchSize,
    lastRunSignature,
    setLastRunSignature,
  } = slotState;
  const [renamingSlotId, setRenamingSlotId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const skipRenameBlur = useRef(false);

  function startRenameSlot(slot: (typeof slots)[number]) {
    setRenamingSlotId(slot.id);
    setRenameDraft(slotLabelFor(slot));
  }
  function commitRenameSlot() {
    if (renamingSlotId) renameSlot(renamingSlotId, renameDraft);
    setRenamingSlotId(null);
  }
  function cancelRenameSlot() {
    skipRenameBlur.current = true;
    setRenamingSlotId(null);
  }

  const [queueCollapsed, setQueueCollapsed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A short confirmation under the buttons (pinned, or a recall that had to reuse this tab).
  const [notice, setNotice] = useState<string | null>(null);
  // The recall requests already acted on, so one request can never open two tabs (the effects below
  // re-run on every render until App has cleared the request).
  const handledRecall = useRef<GenerationRecord | null>(null);
  const handledPrompt = useRef(false);

  const { showRecord } = queue;
  const runningJob = queue.jobs.find((j) => j.status === 'running');
  const busy = queue.jobs.some((j) => j.status === 'queued' || j.status === 'running');
  // Each working tab shows its own results: what finishes for another tab never replaces them.
  const shownBatch = queue.viewBatchFor(activeSlotId);
  const viewSlots = shownBatch === null ? [] : queue.jobs.filter((j) => j.batchId === shownBatch);

  // Estimated time for the settings currently in the form, from earlier runs' timings.
  const [currentEstimate, setCurrentEstimate] = useState<TimeEstimate | null>(null);
  const finishedRuns = queue.jobs.filter((j) => j.status === 'done').length;
  // What will run just before a new job: the last thing waiting, or (undefined) whatever ran last.
  const lastActiveFamily = [...queue.jobs].reverse().find((j) => j.status === 'queued' || j.status === 'running')?.family;
  // Several at once need different seeds, so a locked seed always means exactly one.
  const effectiveBatch = seedLocked ? 1 : batchSize;
  // Video has no steps/CFG fields of its own - the Quality choice decides both.
  const runSteps = mode === 'video' ? VIDEO_QUALITY_SETTINGS[videoQuality].steps : steps;
  const runCfg = mode === 'video' ? VIDEO_QUALITY_SETTINGS[videoQuality].cfg : cfg;

  // Everything that decides what a run produces. The same signature with the same seed is the same
  // picture, so a locked seed plus an unchanged signature would only repeat the last result.
  function runSignature(seedValue: number): string {
    return JSON.stringify([
      mode,
      prompt.trim(),
      width,
      height,
      seedValue,
      mode === 'image' ? [steps, cfg] : [secondsToFrames(lengthSeconds), videoQuality, sourceImagePath],
    ]);
  }
  const repeatsLastRun = seedLocked && lastRunSignature === runSignature(seed);

  function handleModeChange(newMode: Mode) {
    setMode(newMode);
    setCustomSize(false);
    setSourceImagePath(null);
    if (newMode === 'video') {
      setWidth(640);
      setHeight(640);
    } else {
      setWidth(1024);
      setHeight(1024);
    }
  }

  /** Switch to video mode with `request.imagePath` as the source image. The video size keeps
   * the image's aspect ratio (long side VIDEO_LONG_SIDE, both sides a multiple of 16 as Wan
   * needs) rather than the square default handleModeChange would set. */
  function setUpVideoFromImage(request: VideoSourceRequest) {
    const scale = VIDEO_LONG_SIDE / Math.max(request.width, request.height);
    const snap = (n: number) => Math.max(256, Math.round((n * scale) / 16) * 16);
    setMode('video');
    setWidth(snap(request.width));
    setHeight(snap(request.height));
    setSourceImagePath(request.imagePath);
    setError(null);
  }

  useEffect(() => {
    if (!videoSource) return;
    setUpVideoFromImage(videoSource);
    onVideoSourceHandled();
  }, [videoSource, onVideoSourceHandled]);

  async function handleChooseSourceImage() {
    const path = await window.kvgenius.chooseSourceImage();
    if (path) setSourceImagePath(path);
  }

  /** Loads a past generation's prompt and exact settings into the form and shows it in the viewer,
   * under working tab `slotId` (the form being edited must already be that tab's). */
  const applyRecord = useCallback(
    (record: GenerationRecord, slotId: string) => {
      const recalledMode: Mode = FAMILY_KIND[record.modelFamily] === 'video' ? 'video' : 'image';
      setMode(recalledMode);
      setPrompt(record.prompt);
      setWidth(record.width);
      setHeight(record.height);
      setSeed(record.seed);
      setSeedLocked(true);
      setSteps(record.steps);
      setCfg(record.cfg);
      setLengthSeconds(record.length ? framesToSeconds(record.length) : 5);
      setVideoQuality(videoQualityFromCfg(record.cfg));
      // A video keeps a copy of the image it was made from, so it can be re-run in place. Videos made
      // before that was kept have none: a new one has to be chosen before they can be re-run.
      setSourceImagePath(recalledMode === 'video' ? record.sourceImagePath : null);
      showRecord(record, recalledMode, window.kvgenius.imageUrlFor(record.imagePath), slotId);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showRecord]
  );

  /** "Re-rack": opens a past generation (prompt, settings and picture) in a new working tab, so
   * what is in the current tab is left alone. With every tab in use it reuses the current one. */
  function rerackInNewTab(record: GenerationRecord) {
    const slotId = addSlot();
    if (slotId === null) {
      setNotice(`All ${MAX_PROMPT_SLOTS} tabs are in use - loaded into this one.`);
      setTimeout(() => setNotice(null), 4000);
    }
    applyRecord(record, slotId ?? activeSlotId);
  }

  /** A prompt picked in Library > Prompts goes into a new tab too (or this one, if all are in use). */
  function openPromptInNewTab(text: string) {
    if (addSlot() === null) {
      setNotice(`All ${MAX_PROMPT_SLOTS} tabs are in use - loaded into this one.`);
      setTimeout(() => setNotice(null), 4000);
    }
    setPrompt(text);
  }

  useEffect(() => {
    if (!recallRecord) {
      handledRecall.current = null;
      return;
    }
    if (handledRecall.current === recallRecord) return;
    handledRecall.current = recallRecord;
    rerackInNewTab(recallRecord);
    onRecalled();
  }, [recallRecord, onRecalled]);

  useEffect(() => {
    if (recallPrompt === null) {
      handledPrompt.current = false;
      return;
    }
    if (handledPrompt.current) return;
    handledPrompt.current = true;
    openPromptInNewTab(recallPrompt);
    onPromptRecalled();
  }, [recallPrompt, onPromptRecalled]);

  useEffect(() => {
    const timer = setTimeout(() => {
      window.kvgenius
        .estimateGeneration(
          FAMILY_FOR_MODE[mode],
          {
            prompt: '',
            width,
            height,
            seed: 0,
            steps: runSteps,
            cfg: runCfg,
            ...(mode === 'video' ? { length: secondsToFrames(lengthSeconds) } : {}),
          },
          lastActiveFamily
        )
        .then(setCurrentEstimate)
        .catch(() => setCurrentEstimate(null));
    }, 250);
    return () => clearTimeout(timer);
  }, [mode, width, height, runSteps, runCfg, lengthSeconds, lastActiveFamily, finishedRuns]);

  const sizePresets = mode === 'image' ? IMAGE_SIZE_PRESETS : VIDEO_SIZE_PRESETS;
  const presetIndex = sizePresets.findIndex((preset) => preset.width === width && preset.height === height);
  // A size that matches no preset (recalled from the library, from a source image) is shown as custom.
  const sizeSelectValue = customSize || presetIndex < 0 ? 'custom' : String(presetIndex);

  let estimateText = 'No time estimate yet - it learns from your generations.';
  if (currentEstimate) {
    const first = currentEstimate.totalMs;
    // After the first of a batch the models are loaded, so the rest only take the generating time.
    const all = first + (effectiveBatch - 1) * currentEstimate.generateMs;
    estimateText = `Estimated time: ${formatEstimate(effectiveBatch > 1 ? all : first)}${effectiveBatch > 1 ? ` for ${effectiveBatch}` : ''}`;
    if (currentEstimate.loadMs && currentEstimate.loadMs >= 2000) {
      estimateText += ` (includes about ${formatDuration(currentEstimate.loadMs)} loading models)`;
    }
  }

  function handleGenerate() {
    if (!prompt.trim()) {
      setError('Enter a prompt first.');
      return;
    }
    if (mode === 'video' && !sourceImagePath) {
      setError('Choose a source image first.');
      return;
    }
    if (repeatsLastRun) return;
    setError(null);

    const seeds = seedLocked ? [seed] : uniqueRandomSeeds(effectiveBatch);
    if (!seedLocked) setSeed(seeds[0]);
    const base = {
      prompt,
      width,
      height,
      steps: runSteps,
      cfg: runCfg,
      ...(mode === 'video' ? { length: secondsToFrames(lengthSeconds), sourceImagePath: sourceImagePath ?? undefined } : {}),
    };
    const added = queue.enqueue(
      seeds.map((jobSeed) => ({ family: FAMILY_FOR_MODE[mode], kind: mode, params: { ...base, seed: jobSeed } })),
      activeSlotId
    );
    if (added < seeds.length) {
      setError(`The queue is full (${MAX_PENDING_JOBS} waiting) - added ${added} of ${seeds.length}.`);
    }
    // The seed box shows the first seed of the run, which is what locking it later would reuse.
    if (added > 0) setLastRunSignature(runSignature(seeds[0]));
  }

  function handleCancel() {
    if (runningJob) queue.cancelJob(runningJob.id);
  }

  async function handleToggleFavorite(record: GenerationRecord) {
    const favorite = !record.favorite;
    try {
      // Favoriting moves the file into the favorites folder (and back), so its path can change.
      const { imagePath } = await window.kvgenius.setGenerationFavorite(record.id, favorite);
      queue.updateRecord(record.id, { favorite });
      queue.relocateFile(record.id, record.imagePath, imagePath, window.kvgenius.imageUrlFor(imagePath));
      setSourceImagePath((current) => (current === record.imagePath ? imagePath : current));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  /** Pins (or unpins) a result as the example of its prompt - the Library > Prompts gallery. */
  async function handleTogglePinned(record: GenerationRecord) {
    const pinned = !record.pinned;
    try {
      await window.kvgenius.setGenerationPinned(record.id, pinned);
      queue.updateRecord(record.id, { pinned });
      setNotice(pinned ? 'Pinned - find it under Library > Prompts.' : null);
      setTimeout(() => setNotice(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteResult(record: GenerationRecord) {
    const note = (record.favorite ? ' It is marked as a favorite.' : '') + (record.pinned ? ' It is pinned under Prompts.' : '');
    if (!window.confirm(`Delete this generation? This removes the file from disk too.${note}`)) return;
    try {
      await window.kvgenius.deleteGeneration(record.id, record.imagePath);
      queue.removeRecord(record.id);
      // If it was the Source Image for a video, that file is gone.
      setSourceImagePath((current) => (current === record.imagePath ? null : current));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleConvertToVideo(record: GenerationRecord) {
    setUpVideoFromImage({ imagePath: record.imagePath, width: record.width, height: record.height });
  }

  return (
    <div className="page generate-page">
      <div className={`generate-layout${queueCollapsed ? ' generate-layout--queue-collapsed' : ''}`}>
        <div className="generate-sidebar-group">
          <div className="prompt-slots-rail" role="tablist">
            <div className="prompt-slots-rail__list">
              {slots.map((slot) => (
                <div
                  key={slot.id}
                  className={`prompt-slot-row${slot.id === activeSlotId ? ' prompt-slot-row--active' : ''}`}
                >
                  {renamingSlotId === slot.id ? (
                    <input
                      autoFocus
                      className="prompt-slot-row__rename"
                      value={renameDraft}
                      onChange={(e) => setRenameDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          commitRenameSlot();
                        } else if (e.key === 'Escape') {
                          e.preventDefault();
                          cancelRenameSlot();
                        }
                      }}
                      onBlur={() => {
                        if (skipRenameBlur.current) {
                          skipRenameBlur.current = false;
                          return;
                        }
                        commitRenameSlot();
                      }}
                    />
                  ) : (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={slot.id === activeSlotId}
                      className="prompt-slot-row__label"
                      onClick={() => switchSlot(slot.id)}
                      onDoubleClick={() => startRenameSlot(slot)}
                      title={`${slotLabelFor(slot)} (double-click to rename)`}
                    >
                      {slotLabelFor(slot)}
                    </button>
                  )}
                  {slots.length > 1 && (
                    <button
                      type="button"
                      className="prompt-slot-row__close"
                      onClick={() => {
                        closeSlot(slot.id);
                        queue.forgetSlot(slot.id);
                      }}
                      title="Close this tab"
                    >
                      ✕
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button
              type="button"
              className="prompt-slot-row__add"
              onClick={addSlot}
              disabled={slots.length >= MAX_PROMPT_SLOTS}
              title={slots.length >= MAX_PROMPT_SLOTS ? `Up to ${MAX_PROMPT_SLOTS} tabs` : 'New prompt tab'}
            >
              ＋ New
            </button>
          </div>

          <div className="generate-form">
          <div className="button-row--even" style={{ marginBottom: 12 }}>
            <button
              type="button"
              className={mode === 'image' ? 'primary' : undefined}
              onClick={() => handleModeChange('image')}
            >
              🖼️ Image
            </button>
            <button
              type="button"
              className={mode === 'video' ? 'primary' : undefined}
              onClick={() => handleModeChange('video')}
            >
              🎬 Video
            </button>
          </div>

          {mode === 'video' && (
            <div style={{ marginBottom: 12 }}>
              <label className="field-label">Source Image</label>
              <button
                type="button"
                className="source-image-button"
                onClick={handleChooseSourceImage}
                title={sourceImagePath ?? undefined}
              >
                <span className="source-image-button__name">
                  {sourceImagePath ? sourceImagePath.split(/[\\/]/).pop() : 'Choose Source Image...'}
                </span>
              </button>
              {sourceImagePath && (
                <div className="source-image-preview-wrap">
                  <ExpandButton src={window.kvgenius.imageUrlFor(sourceImagePath)} kind="image" alt="Source image" />
                  <img
                    className="source-image-preview"
                    src={window.kvgenius.imageUrlFor(sourceImagePath)}
                    alt="Source image"
                  />
                </div>
              )}
            </div>
          )}

          <div className="field-label-row">
            <label className="field-label" htmlFor="prompt">
              Prompt
            </label>
            <CopyButton text={prompt} title="Copy the prompt" />
          </div>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the image you want..."
            style={{ width: '100%', flex: 1, minHeight: 80, resize: 'none' }}
          />

          <div style={{ marginTop: 12 }}>
            <label className="field-label" htmlFor="size-preset">
              Size
            </label>
            <select
              id="size-preset"
              value={sizeSelectValue}
              onChange={(e) => {
                if (e.target.value === 'custom') {
                  setCustomSize(true);
                  return;
                }
                const preset = sizePresets[Number(e.target.value)];
                if (!preset) return;
                setCustomSize(false);
                setWidth(preset.width);
                setHeight(preset.height);
              }}
              style={{ width: '100%' }}
            >
              {sizePresets.map((preset, i) => (
                <option key={preset.label} value={i}>
                  {preset.label} - {preset.width}×{preset.height}
                </option>
              ))}
              <option value="custom">Custom size...</option>
            </select>
          </div>

          {sizeSelectValue === 'custom' && (
            <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <label className="field-label" htmlFor="width">
                  Width
                </label>
                <input
                  id="width"
                  type="number"
                  value={width}
                  step={mode === 'video' ? 16 : 64}
                  min={256}
                  onChange={(e) => setWidth(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <label className="field-label" htmlFor="height">
                  Height
                </label>
                <input
                  id="height"
                  type="number"
                  value={height}
                  step={mode === 'video' ? 16 : 64}
                  min={256}
                  onChange={(e) => setHeight(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}

          {mode === 'video' && (
            <div style={{ marginTop: 12 }}>
              <label className="field-label" htmlFor="length">
                Length (seconds)
              </label>
              <input
                id="length"
                type="number"
                value={lengthSeconds}
                min={1}
                max={12}
                step={0.5}
                onChange={(e) => setLengthSeconds(Number(e.target.value))}
                style={{ width: 160 }}
              />
              <p style={{ color: 'var(--color-text-muted)', fontSize: 12, marginTop: 4, marginBottom: 0 }}>
                {secondsToFrames(lengthSeconds)} frames @ {VIDEO_FPS}fps. Default is 5s; longer clips take much longer to render.
              </p>

              <label className="field-label" htmlFor="video-quality" style={{ marginTop: 12 }}>
                Quality
              </label>
              <select
                id="video-quality"
                value={videoQuality}
                onChange={(e) => setVideoQuality(e.target.value as VideoQuality)}
                style={{ width: 260 }}
              >
                <option value="fast">Fast (4 steps)</option>
                <option value="high">High (20 steps, slower)</option>
              </select>
              <p style={{ color: 'var(--color-text-muted)', fontSize: 12, marginTop: 4, marginBottom: 0 }}>
                Fast uses a speed-up LoRA and can look grainy. High is cleaner but takes several times longer.
              </p>
            </div>
          )}

          <div style={{ marginTop: 12 }}>
            <label className="field-label" htmlFor="seed">
              Seed
            </label>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                id="seed"
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                style={{ width: 160 }}
              />
              <button
                type="button"
                onClick={() => setSeedLocked((v) => !v)}
                title={seedLocked ? 'Seed is locked - won\'t change between runs' : 'Seed randomizes on each run'}
              >
                {seedLocked ? '🔒 Locked' : '🎲 Random'}
              </button>
            </div>
          </div>

          <div style={{ marginTop: 12 }} className={seedLocked ? 'field--disabled' : undefined}>
            <label className="field-label" htmlFor="batch-size">
              Batch size
            </label>
            <input
              id="batch-size"
              type="number"
              min={1}
              max={MAX_BATCH_SIZE}
              value={effectiveBatch}
              disabled={seedLocked}
              onChange={(e) => setBatchSize(Math.min(MAX_BATCH_SIZE, Math.max(1, Math.floor(Number(e.target.value)) || 1)))}
              style={{ width: 100 }}
            />
            <p style={{ color: 'var(--color-text-muted)', fontSize: 12, marginTop: 4, marginBottom: 0 }}>
              {seedLocked
                ? 'Switch the seed to 🎲 Random to make several at once - each needs its own seed.'
                : `Queues this many, each with its own random seed (up to ${MAX_BATCH_SIZE}).`}
            </p>
          </div>

          {mode === 'image' && (
          <div style={{ marginTop: 16 }}>
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              style={{ width: '100%', justifyContent: 'flex-start' }}
            >
              {advancedOpen ? '▾' : '▸'} Advanced
            </button>
            {advancedOpen && (
              <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
                <div>
                  <label className="field-label" htmlFor="steps">
                    Steps
                  </label>
                  <input
                    id="steps"
                    type="number"
                    value={steps}
                    min={1}
                    max={20}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    style={{ width: 100 }}
                  />
                </div>
                <div>
                  <label className="field-label" htmlFor="cfg">
                    CFG
                  </label>
                  <input
                    id="cfg"
                    type="number"
                    value={cfg}
                    min={0.5}
                    max={3}
                    step={0.1}
                    onChange={(e) => setCfg(Number(e.target.value))}
                    style={{ width: 100 }}
                  />
                </div>
              </div>
            )}
          </div>
          )}

          <div className="generate-actions">
            <button
              type="button"
              className="primary generate-actions__go"
              onClick={handleGenerate}
              disabled={(mode === 'video' && !sourceImagePath) || repeatsLastRun}
              title={repeatsLastRun ? 'Nothing has changed since the last run and the seed is locked' : undefined}
            >
              {busy ? '＋ Queue Another' : 'Generate'}
              {effectiveBatch > 1 ? ` (${effectiveBatch})` : ''}
            </button>
            {runningJob && (
              <button type="button" onClick={handleCancel}>
                ✕ Cancel ({formatElapsed(Math.floor((queue.now - (runningJob.startedAt ?? queue.now)) / 1000))})
              </button>
            )}
          </div>

          <p className="generate-estimate">{estimateText}</p>

          {repeatsLastRun && (
            <p className="generate-repeat-hint">
              Nothing has changed since the last run and the seed is locked, so it would make the exact same result.
              Change a setting, or switch the seed to 🎲 Random.
            </p>
          )}
          {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
          {notice && <p style={{ color: 'var(--color-accent-green)' }}>{notice}</p>}
          </div>
        </div>

        <div className="generate-preview">
          {/* Keyed by tab so switching starts the viewer fresh instead of carrying the previous tab's picture over. */}
          <ResultViewer
            key={activeSlotId}
            slots={viewSlots}
            now={queue.now}
            progressInfo={queue.progressInfo}
            onDelete={handleDeleteResult}
            onToggleFavorite={handleToggleFavorite}
            onTogglePinned={handleTogglePinned}
            onConvertToVideo={handleConvertToVideo}
            onCancelJob={queue.cancelJob}
          />
        </div>

        <QueuePanel
          jobs={queue.jobs}
          now={queue.now}
          progressInfo={queue.progressInfo}
          collapsed={queueCollapsed}
          onToggle={() => setQueueCollapsed((v) => !v)}
          onCancelJob={queue.cancelJob}
          onClearQueued={queue.clearQueued}
          onDismissFailed={queue.dismissFailed}
          onToggleFavorite={handleToggleFavorite}
          onRerack={rerackInNewTab}
        />
      </div>

    </div>
  );
}
