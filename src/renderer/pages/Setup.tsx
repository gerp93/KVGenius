import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { manifestFeature } from '../../shared/modelManifest';
import { summarize } from '../../shared/modelStatus';
import ModelFilesTable from '../components/ModelFilesTable';
import Stepper, { ExternalLink, Step } from '../components/Stepper';
import { useModelStatus } from '../hooks/useModelStatus';

const POLL_MS = 3000;

/** True once ComfyUI answers at the configured address; rechecked every few seconds, so the guide
 * notices by itself when the reader gets it running. */
function useComfyConnected(): boolean | null {
  const [connected, setConnected] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    async function check() {
      const reachable = await window.kvgenius.checkComfyUIConnection().catch(() => false);
      if (!cancelled) setConnected(reachable);
    }
    void check();
    const interval = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);
  return connected;
}

function Callout({ children }: { children: React.ReactNode }) {
  return <div className="stepper__callout">{children}</div>;
}

const STATUS_LABEL = {
  null: '⏳ Checking...',
  true: '🟢 ComfyUI is connected',
  false: '🔴 ComfyUI is not reachable yet',
} as const;

/** A step-by-step walk through getting ComfyUI ready for KVGenius: install it, add the models the
 * templates in src/main/templates expect (file names must match those templates), and connect. */
export default function Setup() {
  const navigate = useNavigate();
  const connected = useComfyConnected();
  const { report } = useModelStatus();
  const [modelsDir, setModelsDir] = useState<string | null>(null);
  useEffect(() => {
    void window.kvgenius.getModelsDirInfo().then((info) => setModelsDir(info.valid ? info.effective : null));
  }, []);
  const zImage = manifestFeature('z-image')!;
  const wan = manifestFeature('wan22-i2v')!;
  const upscaleModels = report && report.source !== 'none' ? report.installed.upscale_models : null;
  // How many of the picture files are still missing, for the connect step (null when nothing is known).
  const missingPictureFiles = report && report.source !== 'none' ? zImage.files.length - summarize(zImage.files, report).present : null;

  const steps: Step[] = [
    {
      kicker: 'Overview',
      title: 'What you are setting up',
      body: (
        <>
          <p>
            KVGenius is the front end. The pictures and videos are made by <strong>ComfyUI</strong>, a free program that
            runs AI models on your own computer - KVGenius sends it a request, waits, and keeps the result in your
            Library. Nothing is sent to an online service.
          </p>
          <p>Getting there takes four things, in this order:</p>
          <ol>
            <li>Install ComfyUI.</li>
            <li>Download the model files for what you want to make (pictures, and optionally video and upscaling).</li>
            <li>Start ComfyUI.</li>
            <li>Tell KVGenius where it is (usually nothing to do - the default just works for ComfyUI Desktop).</li>
          </ol>
          <Callout>
            A computer with a recent graphics card (NVIDIA is the best supported) and plenty of free disk space is
            needed. Model files are several GB each, and video needs far more of both than pictures do.
          </Callout>
        </>
      ),
    },
    {
      kicker: 'Step 1',
      title: 'Install ComfyUI',
      body: (
        <>
          <p>Pick one - both work with KVGenius.</p>
          <ul>
            <li>
              <strong>ComfyUI Desktop</strong> (easiest): a normal installer with a shortcut. Available for Windows,
              macOS (Apple Silicon) and Linux. KVGenius finds it automatically and its default address is the one
              KVGenius expects (port <code>8000</code>).
            </li>
            <li>
              <strong>Portable / manual install</strong>: a folder you unzip (Windows) or set up yourself. It listens on
              port <code>8188</code> instead, and you start it with its run script, for example{' '}
              <code>run_nvidia_gpu.bat</code>. You will set that address and script in a later step.
            </li>
          </ul>
          <p>Run it once after installing so it can finish its own first-time setup.</p>
        </>
      ),
      links: [
        { label: 'Download ComfyUI', href: 'https://www.comfy.org/download' },
        { label: 'Desktop install guide', href: 'https://docs.comfy.org/installation/desktop/windows' },
        { label: 'Portable (Windows) guide', href: 'https://docs.comfy.org/installation/comfyui_portable_windows' },
      ],
    },
    {
      kicker: 'Step 2',
      title: 'Download the picture models',
      body: (
        <>
          <p>
            Text to image uses <strong>Z Image Turbo</strong>. Download these three files and put each one in the folder
            shown. The <strong>names must match exactly</strong> - KVGenius asks ComfyUI for these file names.
          </p>
          <ModelFilesTable feature={zImage} report={report} modelsDir={modelsDir} />
          <p>
            ComfyUI's Z Image Turbo page has the download links. The <code>models</code> folder is inside your ComfyUI
            folder (for ComfyUI Desktop, the base folder you picked when installing). The Status column updates by itself
            as you add the files.
          </p>
          <Callout>
            If ComfyUI was already running, restart it (or press <code>R</code> in its window) so it notices the new
            files.
          </Callout>
        </>
      ),
      links: [{ label: zImage.source.label, href: zImage.source.url }, { label: 'Check all model files', to: '/models' }],
    },
    {
      kicker: 'Optional',
      title: 'Add video (Wan 2.2)',
      body: (
        <>
          <p>
            Skip this if you only want pictures. Making a video from a picture uses <strong>Wan 2.2 image to video</strong>{' '}
            (14B). It needs all six of these:
          </p>
          <ModelFilesTable feature={wan} report={report} modelsDir={modelsDir} />
          <p>
            Two of them are the 4-step LoRAs behind the <strong>Fast</strong> quality option. The workflow contains both
            LoRAs, so install all six even if you only plan to use High. They are all in the Wan 2.2 repackaged
            repository, under its <code>split_files</code> folder.
          </p>
          <Callout>
            Video is much heavier than pictures: expect large downloads, a lot of graphics memory, and slow runs.
            ComfyUI's Wan 2.2 page lists the requirements.
          </Callout>
        </>
      ),
      links: [{ label: wan.source.label, href: wan.source.url }],
    },
    {
      kicker: 'Optional',
      title: 'Add an upscale model',
      body: (
        <>
          <p>
            Skip this if you do not plan to enlarge pictures or videos. Tools &gt; Upscale and the details panel's Upscale
            button use an upscale model of your choice. Any ESRGAN-style model works - for example 4x-UltraSharp or
            RealESRGAN x4 - and several can be installed side by side.
          </p>
          <p>
            Put the downloaded file (<code>.pth</code> or <code>.safetensors</code>) in <code>upscale_models</code>.
          </p>
          {upscaleModels !== null && (
            <p>
              {upscaleModels.length > 0
                ? `Found right now: ${upscaleModels.join(', ')}.`
                : 'No upscale models found yet.'}
            </p>
          )}
        </>
      ),
      links: [{ label: 'Browse upscale models', href: manifestFeature('upscale')!.source.url }],
    },
    {
      kicker: 'Step 3',
      title: 'Start ComfyUI and connect',
      body: (
        <>
          <div className="stepper__status" role="status">
            {STATUS_LABEL[String(connected) as keyof typeof STATUS_LABEL]}
          </div>
          {connected ? (
            <>
              <p>KVGenius can reach ComfyUI.</p>
              {missingPictureFiles === null ? null : missingPictureFiles === 0 ? (
                <p>All the picture model files are in place. You are ready to make something.</p>
              ) : (
                <p>
                  {missingPictureFiles} of the {zImage.files.length} picture model files are still missing - go back to step 2 to
                  see which.
                </p>
              )}
            </>
          ) : (
            <>
              <p>
                Start ComfyUI the way you normally would, or click the red <strong>ComfyUI not reachable</strong>{' '}
                indicator at the top right of KVGenius and it will start it for you. This page notices when it comes
                up, which can take a minute the first time while models load.
              </p>
              <h4>Still red?</h4>
              <ul>
                <li>
                  <strong>Check the address.</strong> ComfyUI Desktop uses <code>http://localhost:8000</code>; a
                  portable or manual install usually uses <code>http://localhost:8188</code>. Set it in Settings &gt;
                  ComfyUI.
                </li>
                <li>
                  <strong>Set the launch shortcut.</strong> For a portable install, choose its run script (such as{' '}
                  <code>run_nvidia_gpu.bat</code>) in Settings &gt; ComfyUI &gt; Launch shortcut so KVGenius can start it.
                </li>
                <li>
                  <strong>ComfyUI's own window</strong> shows any error that stopped it from starting.
                </li>
              </ul>
            </>
          )}
        </>
      ),
      links: [
        { label: 'Open Settings > ComfyUI', to: '/settings?tab=comfyui' },
        { label: 'Check all model files', to: '/models' },
      ],
    },
    {
      kicker: 'Done',
      title: 'Make your first picture',
      body: (
        <>
          <p>
            Go to <strong>Generate</strong>, type a prompt, and press Generate. The job appears in the queue bar along the
            bottom, and the result lands in <strong>Library &gt; Output</strong>.
          </p>
          <ul>
            <li>
              <strong>Nothing happens or it fails?</strong> The most common cause is a model file that is missing or
              named differently from the tables in this guide.
            </li>
            <li>
              <strong>Want video?</strong> Open any picture's details and press 🎬 Video (needs the Wan 2.2 files).
            </li>
            <li>
              You can come back to this guide any time from Settings &gt; ComfyUI, or from the Setup guide link that
              appears next to the ComfyUI indicator whenever it cannot be reached.
            </li>
          </ul>
          <p>
            More about ComfyUI itself: <ExternalLink href="https://docs.comfy.org">docs.comfy.org</ExternalLink>
          </p>
        </>
      ),
    },
  ];

  return (
    <div className="page">
      <Stepper
        title="Set up ComfyUI"
        subtitle="Get ComfyUI installed, add the models, and connect it to KVGenius"
        steps={steps}
        finishLabel="Go to Generate"
        onFinish={() => navigate('/')}
      />
    </div>
  );
}
