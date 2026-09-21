import { useEffect, useState } from 'react';
import PromptModal from '../components/PromptModal';
import QueuePanel from '../components/QueuePanel';
import ResultViewer from '../components/ResultViewer';
import ExpandButton from '../components/Lightbox';
import { MAX_BATCH_SIZE, MAX_PENDING_JOBS, useGenerationQueue } from '../hooks/useGenerationQueue';
import { formatDuration, formatElapsed, formatEstimate } from '../utils/format';
import { VIDEO_FPS, framesToSeconds, secondsToFrames } from '../utils/video';
import { FAMILY_KIND, GenerationRecord, TimeEstimate, VideoSourceRequest } from '../../shared/types';

type Mode = 'image' | 'video';

const FAMILY_FOR_MODE: Record<Mode, string> = {
  image: 'z-image-turbo',
  video: 'wan22-i2v',
};

interface Props {
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
  recallRecord,
  onRecalled,
  recallPrompt,
  onPromptRecalled,
  videoSource,
  onVideoSourceHandled,
}: Props) {
  const [mode, setMode] = useState<Mode>('image');
  const [prompt, setPrompt] = useState('');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState<number>(randomSeed());
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(8);
  const [cfg, setCfg] = useState(1);
  const [lengthSeconds, setLengthSeconds] = useState(5);
  const [sourceImagePath, setSourceImagePath] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Picking "Custom size..." from the dropdown reveals the width/height boxes.
  const [customSize, setCustomSize] = useState(false);

  const [batchSize, setBatchSize] = useState(1);
  const [queueCollapsed, setQueueCollapsed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  // Save Prompt opens a modal asking for a name and optional tags.
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [existingTags, setExistingTags] = useState<string[]>([]);

  const queue = useGenerationQueue();
  const { showRecord } = queue;
  const runningJob = queue.jobs.find((j) => j.status === 'running');
  const busy = queue.jobs.some((j) => j.status === 'queued' || j.status === 'running');
  const viewSlots = queue.jobs.filter((j) => j.batchId === queue.viewBatchId);

  // Estimated time for the settings currently in the form, from earlier runs' timings.
  const [currentEstimate, setCurrentEstimate] = useState<TimeEstimate | null>(null);
  const finishedRuns = queue.jobs.filter((j) => j.status === 'done').length;
  // What will run just before a new job: the last thing waiting, or (undefined) whatever ran last.
  const lastActiveFamily = [...queue.jobs].reverse().find((j) => j.status === 'queued' || j.status === 'running')?.family;
  // Several at once need different seeds, so a locked seed always means exactly one.
  const effectiveBatch = seedLocked ? 1 : batchSize;

  // Everything that decides what a run produces. The same signature with the same seed is the same
  // picture, so a locked seed plus an unchanged signature would only repeat the last result.
  function runSignature(seedValue: number): string {
    return JSON.stringify([
      mode,
      prompt.trim(),
      width,
      height,
      seedValue,
      mode === 'image' ? [steps, cfg] : [secondsToFrames(lengthSeconds), sourceImagePath],
    ]);
  }
  const [lastRunSignature, setLastRunSignature] = useState<string | null>(null);
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

  useEffect(() => {
    if (!recallRecord) return;
    const recalledMode: Mode = FAMILY_KIND[recallRecord.modelFamily] === 'video' ? 'video' : 'image';
    setMode(recalledMode);
    setPrompt(recallRecord.prompt);
    setWidth(recallRecord.width);
    setHeight(recallRecord.height);
    setSeed(recallRecord.seed);
    setSeedLocked(true);
    setSteps(recallRecord.steps);
    setCfg(recallRecord.cfg);
    setLengthSeconds(recallRecord.length ? framesToSeconds(recallRecord.length) : 5);
    // The source image used for a past video generation isn't retained - only the
    // resulting video is. A new one has to be chosen before this can be re-run.
    setSourceImagePath(null);
    showRecord(recallRecord, recalledMode, window.kvgenius.imageUrlFor(recallRecord.imagePath));
    onRecalled();
  }, [recallRecord, onRecalled, showRecord]);

  useEffect(() => {
    if (recallPrompt === null) return;
    setPrompt(recallPrompt);
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
            steps,
            cfg,
            ...(mode === 'video' ? { length: secondsToFrames(lengthSeconds) } : {}),
          },
          lastActiveFamily
        )
        .then(setCurrentEstimate)
        .catch(() => setCurrentEstimate(null));
    }, 250);
    return () => clearTimeout(timer);
  }, [mode, width, height, steps, cfg, lengthSeconds, lastActiveFamily, finishedRuns]);

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
      steps,
      cfg,
      ...(mode === 'video' ? { length: secondsToFrames(lengthSeconds), sourceImagePath: sourceImagePath ?? undefined } : {}),
    };
    const added = queue.enqueue(
      seeds.map((jobSeed) => ({ family: FAMILY_FOR_MODE[mode], kind: mode, params: { ...base, seed: jobSeed } }))
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

  async function handleDeleteResult(record: GenerationRecord) {
    const note = record.favorite ? ' It is marked as a favorite.' : '';
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

  async function openSavePromptModal() {
    if (!prompt.trim()) return;
    try {
      // Tags already in use, offered as suggestions in the modal.
      const saved = await window.kvgenius.listSavedPrompts();
      setExistingTags([...new Set(saved.flatMap((sp) => sp.tags))]);
    } catch {
      setExistingTags([]);
    }
    setSaveModalOpen(true);
  }

  async function handleSavePrompt(name: string, tags: string[]) {
    await window.kvgenius.savePrompt(name, prompt, tags);
    setSaveModalOpen(false);
    setSaveStatus(`Saved "${name}" - find it under Library > Prompts.`);
    setTimeout(() => setSaveStatus(null), 3000);
  }

  return (
    <div className="page generate-page">
      <div className={`generate-layout${queueCollapsed ? ' generate-layout--queue-collapsed' : ''}`}>
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

          <label className="field-label" htmlFor="prompt">
            Prompt
          </label>
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

          <div className={`generate-actions${busy ? ' generate-actions--busy' : ''}`}>
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
            <button type="button" onClick={openSavePromptModal} disabled={!prompt.trim()}>
              Save Prompt
            </button>
          </div>

          <p className="generate-estimate">{estimateText}</p>

          {repeatsLastRun && (
            <p className="generate-repeat-hint">
              Nothing has changed since the last run and the seed is locked, so it would make the exact same result.
              Change a setting, or switch the seed to 🎲 Random.
            </p>
          )}
          {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
          {saveStatus && <p style={{ color: 'var(--color-accent-green)' }}>{saveStatus}</p>}
        </div>

        <div className="generate-preview">
          <ResultViewer
            slots={viewSlots}
            now={queue.now}
            progressInfo={queue.progressInfo}
            onDelete={handleDeleteResult}
            onToggleFavorite={handleToggleFavorite}
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
        />
      </div>

      {saveModalOpen && (
        <PromptModal
          title="Save prompt"
          submitLabel="Save prompt"
          prompt={prompt}
          existingTags={existingTags}
          onSave={handleSavePrompt}
          onClose={() => setSaveModalOpen(false)}
        />
      )}
    </div>
  );
}
