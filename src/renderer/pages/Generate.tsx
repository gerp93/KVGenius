import { useEffect, useState } from 'react';
import { GenerationRecord } from '../../shared/types';

interface Props {
  recallRecord: GenerationRecord | null;
  onRecalled: () => void;
}

function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 32);
}

export default function Generate({ recallRecord, onRecalled }: Props) {
  const [prompt, setPrompt] = useState('');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState<number>(randomSeed());
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(8);
  const [cfg, setCfg] = useState(1);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!recallRecord) return;
    setPrompt(recallRecord.prompt);
    setWidth(recallRecord.width);
    setHeight(recallRecord.height);
    setSeed(recallRecord.seed);
    setSeedLocked(true);
    setSteps(recallRecord.steps);
    setCfg(recallRecord.cfg);
    setImageUrl(window.kvgenius.imageUrlFor(recallRecord.imagePath));
    onRecalled();
  }, [recallRecord, onRecalled]);

  async function handleGenerate() {
    if (!prompt.trim()) {
      setError('Enter a prompt first.');
      return;
    }
    setError(null);
    setIsGenerating(true);
    const usedSeed = seedLocked ? seed : randomSeed();
    if (!seedLocked) setSeed(usedSeed);

    try {
      const result = await window.kvgenius.generate({
        prompt,
        width,
        height,
        seed: usedSeed,
        steps,
        cfg,
      });
      setImageUrl(result.imageUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleSavePrompt() {
    if (!prompt.trim()) return;
    try {
      await window.kvgenius.savePrompt(null, prompt);
      setSaveStatus('Prompt saved.');
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="page generate-page">
      <div className="generate-layout">
        <div className="generate-form">
          <label className="field-label" htmlFor="prompt">
            Prompt
          </label>
          <textarea
            id="prompt"
            rows={5}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the image you want..."
            style={{ width: '100%', resize: 'vertical' }}
          />

          <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
            <div>
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
              />
            </div>
            <div>
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
              />
            </div>
          </div>

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

          <div style={{ marginTop: 16 }}>
            <button type="button" onClick={() => setAdvancedOpen((v) => !v)}>
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

          <div style={{ display: 'flex', gap: 8, marginTop: 20 }}>
            <button type="button" className="primary" onClick={handleGenerate} disabled={isGenerating}>
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
            <button type="button" onClick={handleSavePrompt} disabled={!prompt.trim()}>
              Save Prompt
            </button>
          </div>

          {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
          {saveStatus && <p style={{ color: 'var(--color-accent-green)' }}>{saveStatus}</p>}
        </div>

        <div className="generate-preview">
          {imageUrl ? (
            <img src={imageUrl} alt="Generated" style={{ maxWidth: '100%', borderRadius: 8 }} />
          ) : (
            <div className="generate-preview__placeholder">No image yet</div>
          )}
        </div>
      </div>
    </div>
  );
}
