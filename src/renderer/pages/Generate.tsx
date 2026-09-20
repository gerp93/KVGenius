import { useEffect, useState } from 'react';
import { FAMILY_KIND, GenerationRecord } from '../../shared/types';

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
}

function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 32);
}

const ASPECT_RATIO_PRESETS: { label: string; width: number; height: number }[] = [
  { label: 'Square (1:1)', width: 1024, height: 1024 },
  { label: 'Portrait (2:3)', width: 832, height: 1216 },
  { label: 'Portrait (9:16)', width: 768, height: 1344 },
  { label: 'Landscape (3:2)', width: 1216, height: 832 },
  { label: 'Landscape (16:9)', width: 1344, height: 768 },
];

export default function Generate({ recallRecord, onRecalled, recallPrompt, onPromptRecalled }: Props) {
  const [mode, setMode] = useState<Mode>('image');
  const [prompt, setPrompt] = useState('');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState<number>(randomSeed());
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(8);
  const [cfg, setCfg] = useState(1);
  const [length, setLength] = useState(81);
  const [sourceImagePath, setSourceImagePath] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [resultMode, setResultMode] = useState<Mode>('image');
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!isGenerating) return;
    setElapsedSeconds(0);
    const startedAt = Date.now();
    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isGenerating]);

  function formatElapsed(totalSeconds: number): string {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
  }

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
    setLength(recallRecord.length ?? 81);
    // The source image used for a past video generation isn't retained - only the
    // resulting video is. A new one has to be chosen before this can be re-run.
    setSourceImagePath(null);
    setImageUrl(window.kvgenius.imageUrlFor(recallRecord.imagePath));
    setResultMode(recalledMode);
    onRecalled();
  }, [recallRecord, onRecalled]);

  useEffect(() => {
    if (recallPrompt === null) return;
    setPrompt(recallPrompt);
    onPromptRecalled();
  }, [recallPrompt, onPromptRecalled]);

  async function handleGenerate() {
    if (!prompt.trim()) {
      setError('Enter a prompt first.');
      return;
    }
    if (mode === 'video' && !sourceImagePath) {
      setError('Choose a source image first.');
      return;
    }
    setError(null);
    setIsGenerating(true);
    const usedSeed = seedLocked ? seed : randomSeed();
    if (!seedLocked) setSeed(usedSeed);
    const generatedMode = mode;

    try {
      const result = await window.kvgenius.generate(FAMILY_FOR_MODE[mode], {
        prompt,
        width,
        height,
        seed: usedSeed,
        steps,
        cfg,
        ...(mode === 'video' ? { length, sourceImagePath: sourceImagePath ?? undefined } : {}),
      });
      setImageUrl(result.imageUrl);
      setResultMode(generatedMode);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleCancel() {
    try {
      await window.kvgenius.cancelGeneration();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
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
      <div className="generate-layout">
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
              <button type="button" onClick={handleChooseSourceImage} style={{ width: '100%' }}>
                {sourceImagePath ? sourceImagePath.split('/').pop() : 'Choose Source Image...'}
              </button>
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
                Length (frames)
              </label>
              <input
                id="length"
                type="number"
                value={length}
                min={1}
                max={200}
                onChange={(e) => setLength(Number(e.target.value))}
                style={{ width: 160 }}
              />
              <p style={{ color: 'var(--color-text-muted)', fontSize: 12, marginTop: 4, marginBottom: 0 }}>
                81 frames @ 16fps ≈ 5s (this template's default).
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

          <div className="button-row--even" style={{ marginTop: 20 }}>
            {isGenerating ? (
              <button type="button" onClick={handleCancel}>
                ✕ Cancel ({formatElapsed(elapsedSeconds)})
              </button>
            ) : (
              <button
                type="button"
                className="primary"
                onClick={handleGenerate}
                disabled={mode === 'video' && !sourceImagePath}
              >
                Generate
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
          {isGenerating ? (
            <div className="generate-preview__loading">
              <div className="progress-bar progress-bar--indeterminate" />
              <span>Generating... {formatElapsed(elapsedSeconds)}</span>
              <button type="button" onClick={handleCancel}>
                ✕ Cancel
              </button>
            </div>
          ) : imageUrl && resultMode === 'video' ? (
            <video src={imageUrl} controls style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: 8 }} />
          ) : imageUrl ? (
            <img src={imageUrl} alt="Generated" style={{ maxWidth: '100%', borderRadius: 8 }} />
          ) : (
            <div className="generate-preview__placeholder">No image yet</div>
          )}
        </div>
      </div>
    </div>
  );
}
