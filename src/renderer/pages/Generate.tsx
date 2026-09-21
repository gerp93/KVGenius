import { useEffect, useState } from 'react';
import QueuePanel from '../components/QueuePanel';
import ResultViewer from '../components/ResultViewer';
import ExpandButton from '../components/Lightbox';
import { MAX_BATCH_SIZE, MAX_PENDING_JOBS, useGenerationQueue } from '../hooks/useGenerationQueue';
import { formatElapsed } from '../utils/format';
import { VIDEO_FPS, framesToSeconds, secondsToFrames } from '../utils/video';
import { FAMILY_KIND, GenerationRecord, VideoSourceRequest } from '../../shared/types';

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

const ASPECT_RATIO_PRESETS: { label: string; width: number; height: number }[] = [
  { label: 'Square (1:1)', width: 1024, height: 1024 },
  { label: 'Portrait (2:3)', width: 832, height: 1216 },
  { label: 'Portrait (9:16)', width: 768, height: 1344 },
  { label: 'Landscape (3:2)', width: 1216, height: 832 },
  { label: 'Landscape (16:9)', width: 1344, height: 768 },
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

  const [batchSize, setBatchSize] = useState(1);
  const [queueCollapsed, setQueueCollapsed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const queue = useGenerationQueue();
  const { showRecord } = queue;
  const runningJob = queue.jobs.find((j) => j.status === 'running');
  const busy = queue.jobs.some((j) => j.status === 'queued' || j.status === 'running');
  const viewSlots = queue.jobs.filter((j) => j.batchId === queue.viewBatchId);
  // Several at once need different seeds, so a locked seed always means exactly one.
  const effectiveBatch = seedLocked ? 1 : batchSize;

  function handleModeChange(newMode: Mode) {
    setMode(newMode);
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

  function handleGenerate() {
    if (!prompt.trim()) {
      setError('Enter a prompt first.');
      return;
    }
    if (mode === 'video' && !sourceImagePath) {
      setError('Choose a source image first.');
      return;
    }
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
  }

  function handleCancel() {
    if (runningJob) queue.cancelJob(runningJob.id);
  }

  async function handleToggleFavorite(record: GenerationRecord) {
    const favorite = !record.favorite;
    try {
      await window.kvgenius.setGenerationFavorite(record.id, favorite);
      queue.updateRecord(record.id, { favorite });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleConvertToVideo(record: GenerationRecord) {
    setUpVideoFromImage({ imagePath: record.imagePath, width: record.width, height: record.height });
  }

  async function handleSavePrompt() {
    if (!prompt.trim()) return;
    try {
      await window.kvgenius.savePrompt(null, prompt);
      setSaveStatus('Prompt saved - find it in Library.');
      setTimeout(() => setSaveStatus(null), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
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

          {mode === 'image' && (
            <div style={{ marginTop: 12 }}>
              <label className="field-label" htmlFor="aspect-ratio">
                Aspect Ratio
              </label>
              <select
                id="aspect-ratio"
                defaultValue=""
                onChange={(e) => {
                  const preset = ASPECT_RATIO_PRESETS[Number(e.target.value)];
                  if (!preset) return;
                  setWidth(preset.width);
                  setHeight(preset.height);
                  e.target.value = '';
                }}
                style={{ width: '100%' }}
              >
                <option value="" disabled>
                  Choose a preset...
                </option>
                {ASPECT_RATIO_PRESETS.map((preset, i) => (
                  <option key={preset.label} value={i}>
                    {preset.label} - {preset.width}×{preset.height}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <label className="field-label" htmlFor="width">
                Width
              </label>
              <input
                id="width"
                type="number"
                value={width}
                step={64}
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
                step={64}
                min={256}
                onChange={(e) => setHeight(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>

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

          <div style={{ marginTop: 12 }}>
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
              disabled={mode === 'video' && !sourceImagePath}
            >
              {busy ? '＋ Queue Another' : 'Generate'}
              {effectiveBatch > 1 ? ` (${effectiveBatch})` : ''}
            </button>
            {runningJob && (
              <button type="button" onClick={handleCancel}>
                ✕ Cancel ({formatElapsed(Math.floor((queue.now - (runningJob.startedAt ?? queue.now)) / 1000))})
              </button>
            )}
            <button type="button" onClick={handleSavePrompt} disabled={!prompt.trim()}>
              Save Prompt
            </button>
          </div>

          {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
          {saveStatus && <p style={{ color: 'var(--color-accent-green)' }}>{saveStatus}</p>}
        </div>

        <div className="generate-preview">
          <ResultViewer
            slots={viewSlots}
            now={queue.now}
            onToggleFavorite={handleToggleFavorite}
            onConvertToVideo={handleConvertToVideo}
            onCancelJob={queue.cancelJob}
          />
        </div>

        <QueuePanel
          jobs={queue.jobs}
          now={queue.now}
          collapsed={queueCollapsed}
          onToggle={() => setQueueCollapsed((v) => !v)}
          onCancelJob={queue.cancelJob}
          onClearQueued={queue.clearQueued}
          onDismissFailed={queue.dismissFailed}
        />
      </div>
    </div>
  );
}
