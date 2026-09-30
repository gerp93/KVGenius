import { ChildProcess, execFile, spawn, spawnSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export interface FfmpegPaths {
  ffmpeg: string;
  ffprobe: string;
}

const EXE = process.platform === 'win32' ? '.exe' : '';

function runs(command: string): boolean {
  try {
    return spawnSync(command, ['-version'], { stdio: 'ignore', timeout: 10_000 }).status === 0;
  } catch {
    return false;
  }
}

/** A path inside app.asar can't be executed; electron-builder unpacks binaries next to it. */
function unpacked(p: string): string {
  return p.replace(/app\.asar([\\/])/, 'app.asar.unpacked$1');
}

function optionalModulePath(name: string, pick: (m: unknown) => unknown): string | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const value = pick(require(name));
    return typeof value === 'string' ? unpacked(value) : null;
  } catch {
    return null;
  }
}

/**
 * Finds ffmpeg + ffprobe, in order: the path chosen in Settings (an ffmpeg binary; ffprobe is
 * looked for beside it), the copy bundled through the optional ffmpeg-static / ffprobe-static
 * packages, then whatever is on PATH. Null when none works.
 */
export function findFfmpeg(override?: string | null): FfmpegPaths | null {
  const candidates: Array<() => FfmpegPaths | null> = [];
  if (override && override.trim()) {
    candidates.push(() => {
      const ffmpeg = override.trim();
      if (!fs.existsSync(ffmpeg)) return null;
      const sibling = path.join(path.dirname(ffmpeg), `ffprobe${EXE}`);
      return { ffmpeg, ffprobe: fs.existsSync(sibling) ? sibling : `ffprobe${EXE}` };
    });
  }
  candidates.push(() => {
    const ffmpeg = optionalModulePath('ffmpeg-static', (m) => m);
    const ffprobe = optionalModulePath('ffprobe-static', (m) => (m as { path?: string }).path);
    return ffmpeg && ffprobe && fs.existsSync(ffmpeg) && fs.existsSync(ffprobe) ? { ffmpeg, ffprobe } : null;
  });
  candidates.push(() => ({ ffmpeg: 'ffmpeg', ffprobe: 'ffprobe' }));

  for (const candidate of candidates) {
    const found = candidate();
    if (found && runs(found.ffmpeg) && runs(found.ffprobe)) return found;
  }
  return null;
}

function execBuffer(command: string, args: string[], timeoutMs = 60_000): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    execFile(command, args, { encoding: 'buffer', maxBuffer: 64 * 1024 * 1024, timeout: timeoutMs }, (err, stdout, stderr) => {
      if (err) reject(new Error(`${path.basename(command)} failed: ${stderr.toString().trim().split('\n').slice(-3).join(' ') || err.message}`));
      else resolve(stdout);
    });
  });
}

export interface MediaInfo {
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  hasVideo: boolean;
  hasAudio: boolean;
  videoCodec: string | null;
  audioCodec: string | null;
  format: string | null;
}

interface ProbeStream {
  codec_type?: string;
  codec_name?: string;
  width?: number;
  height?: number;
  avg_frame_rate?: string;
  r_frame_rate?: string;
  duration?: string;
}

function parseRate(rate: string | undefined): number | null {
  if (!rate) return null;
  const [n, d] = rate.split('/').map(Number);
  if (!n || !d) return null;
  return Math.round((n / d) * 1000) / 1000;
}

export async function probeMedia(ff: FfmpegPaths, file: string): Promise<MediaInfo> {
  const out = await execBuffer(ff.ffprobe, ['-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', file]);
  const data = JSON.parse(out.toString('utf-8')) as { streams?: ProbeStream[]; format?: { duration?: string; format_name?: string } };
  const video = data.streams?.find((s) => s.codec_type === 'video');
  const audio = data.streams?.find((s) => s.codec_type === 'audio');
  const rawDuration = Number(data.format?.duration ?? video?.duration ?? audio?.duration);
  return {
    durationSeconds: Number.isFinite(rawDuration) ? Math.round(rawDuration * 1000) / 1000 : null,
    width: video?.width ?? null,
    height: video?.height ?? null,
    fps: parseRate(video?.avg_frame_rate) ?? parseRate(video?.r_frame_rate),
    hasVideo: !!video,
    hasAudio: !!audio,
    videoCodec: video?.codec_name ?? null,
    audioCodec: audio?.codec_name ?? null,
    format: data.format?.format_name ?? null,
  };
}

/** A small JPEG of an image, or of a frame of a video (near the start, where a clip is most
 * recognisable). Null if ffmpeg produced nothing. */
export async function makePoster(ff: FfmpegPaths, file: string, isVideo: boolean, maxSide = 512): Promise<Buffer | null> {
  const scale = `scale='if(gt(iw,ih),min(${maxSide},iw),-2)':'if(gt(iw,ih),-2,min(${maxSide},ih))'`;
  const attempt = (seek: number) =>
    execBuffer(ff.ffmpeg, [
      '-v', 'error',
      ...(isVideo && seek > 0 ? ['-ss', String(seek)] : []),
      '-i', file,
      '-frames:v', '1',
      '-vf', scale,
      '-q:v', '5',
      '-f', 'image2pipe',
      '-vcodec', 'mjpeg',
      'pipe:1',
    ]);
  let bytes = await attempt(isVideo ? 0.5 : 0).catch(() => Buffer.alloc(0));
  if (bytes.length === 0 && isVideo) bytes = await attempt(0).catch(() => Buffer.alloc(0));
  return bytes.length > 0 ? bytes : null;
}

// ---------------------------------------------------------------------------------------------
// Assembling clips + a backing track

export type Transition = 'cut' | 'crossfade';
export type EndBehavior = 'trim_to_video' | 'trim_to_audio' | 'fade_out';

export interface AssembleClipInput {
  path: string;
  /** Probed length of the whole clip. */
  durationSeconds: number;
  /** Use only this much from the start of the clip. */
  trimSeconds?: number | null;
}

export interface AssembleInput {
  clips: AssembleClipInput[];
  /** Output size and frame rate every clip is normalised to (the first clip's). */
  width: number;
  height: number;
  fps: number;
  audio?: { path: string; durationSeconds: number } | null;
  transition: Transition;
  crossfadeSeconds: number;
  end: EndBehavior;
  fadeSeconds: number;
  output: string;
  /** Where the concat list is written when the fast (stream copy) path is used. */
  listFilePath: string;
}

export interface AssemblePlan {
  /** 'copy': clips are joined without re-encoding (fast, lossless). 'reencode': everything is re-encoded. */
  mode: 'copy' | 'reencode';
  args: string[];
  /** Length of the finished video. */
  totalSeconds: number;
  warnings: string[];
  /** Content to write to `listFilePath` before running (copy mode only). */
  listFileContent?: string;
}

const num = (n: number) => String(Math.round(n * 1000) / 1000);

/** Builds the ffmpeg command for an assembly. Pure: no probing, no files - so it can be tested. */
export function planAssemble(input: AssembleInput): AssemblePlan {
  if (input.clips.length === 0) throw new Error('At least one clip is required.');
  const warnings: string[] = [];

  const used = input.clips.map((clip) => {
    const trimmed = clip.trimSeconds != null && clip.trimSeconds > 0 && clip.trimSeconds < clip.durationSeconds;
    return { ...clip, seconds: trimmed ? (clip.trimSeconds as number) : clip.durationSeconds, trimmed };
  });
  const clipTotal = used.reduce((sum, c) => sum + c.seconds, 0);

  let crossfade = 0;
  if (input.transition === 'crossfade' && used.length > 1) {
    const shortest = Math.min(...used.map((c) => c.seconds));
    crossfade = Math.min(Math.max(input.crossfadeSeconds, 0.05), shortest * 0.45);
    if (crossfade < input.crossfadeSeconds) {
      warnings.push(`Crossfade shortened to ${num(crossfade)}s because the shortest clip is ${num(shortest)}s.`);
    }
  }
  const videoSeconds = clipTotal - crossfade * (used.length - 1);

  const audio = input.audio ?? null;
  let total = videoSeconds;
  if (input.end === 'trim_to_audio') {
    if (audio) total = audio.durationSeconds;
    else warnings.push('end="trim_to_audio" has no effect without an audio track.');
  }
  const audioShort = audio !== null && audio.durationSeconds < total - 0.05;
  const videoShort = total > videoSeconds + 0.05;
  if (audioShort) warnings.push(`The audio (${num(audio.durationSeconds)}s) is shorter than the video (${num(total)}s); the end is silent.`);
  if (audio && total < audio.durationSeconds - 0.05) warnings.push(`The audio (${num(audio.durationSeconds)}s) is cut at ${num(total)}s.`);
  if (videoShort) warnings.push(`The clips (${num(videoSeconds)}s) are shorter than the audio; the last frame is held for ${num(total - videoSeconds)}s.`);

  const fade = input.end === 'fade_out' ? Math.min(Math.max(input.fadeSeconds, 0.1), total) : 0;
  const audioFilters = [...(audioShort ? ['apad'] : []), ...(fade > 0 ? [`afade=t=out:st=${num(total - fade)}:d=${num(fade)}`] : [])];
  const progress = ['-progress', 'pipe:1', '-nostats'];
  const audioEncode = ['-c:a', 'aac', '-b:a', '192k'];
  const container = ['-movflags', '+faststart'];

  const canCopy = input.transition === 'cut' && !used.some((c) => c.trimmed) && input.end === 'trim_to_video';
  if (canCopy) {
    const quote = (p: string) => `file '${p.replace(/'/g, "'\\''")}'`;
    const args = [
      '-y', '-v', 'error', ...progress,
      '-f', 'concat', '-safe', '0', '-i', input.listFilePath,
      ...(audio ? ['-i', audio.path] : []),
      '-map', '0:v:0',
      ...(audio ? ['-map', '1:a:0'] : []),
      '-c:v', 'copy',
      ...(audio ? [...audioEncode, ...(audioFilters.length ? ['-af', audioFilters.join(',')] : [])] : ['-an']),
      '-t', num(total),
      ...container,
      input.output,
    ];
    return { mode: 'copy', args, totalSeconds: total, warnings, listFileContent: used.map((c) => quote(c.path)).join('\n') + '\n' };
  }

  const normalise = `fps=${num(input.fps)},scale=${input.width}:${input.height}:force_original_aspect_ratio=decrease,pad=${input.width}:${input.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p`;
  const graph: string[] = used.map(
    (c, i) => `[${i}:v]${c.trimmed ? `trim=duration=${num(c.seconds)},` : ''}setpts=PTS-STARTPTS,${normalise}[v${i}]`
  );
  if (used.length === 1) {
    graph.push('[v0]null[vcat]');
  } else if (crossfade > 0) {
    let elapsed = used[0].seconds;
    let previous = 'v0';
    for (let i = 1; i < used.length; i++) {
      const label = i === used.length - 1 ? 'vcat' : `x${i}`;
      graph.push(`[${previous}][v${i}]xfade=transition=fade:duration=${num(crossfade)}:offset=${num(elapsed - crossfade)}[${label}]`);
      elapsed += used[i].seconds - crossfade;
      previous = label;
    }
  } else {
    graph.push(`${used.map((_, i) => `[v${i}]`).join('')}concat=n=${used.length}:v=1:a=0[vcat]`);
  }
  const tail = [
    ...(videoShort ? [`tpad=stop_mode=clone:stop_duration=${num(total - videoSeconds)}`] : []),
    ...(fade > 0 ? [`fade=t=out:st=${num(total - fade)}:d=${num(fade)}`] : []),
  ];
  graph.push(`[vcat]${tail.length ? tail.join(',') : 'null'}[vout]`);

  const args = [
    '-y', '-v', 'error', ...progress,
    ...used.flatMap((c) => ['-i', c.path]),
    ...(audio ? ['-i', audio.path] : []),
    '-filter_complex', graph.join(';'),
    '-map', '[vout]',
    ...(audio ? ['-map', `${used.length}:a:0`, ...audioEncode, ...(audioFilters.length ? ['-af', audioFilters.join(',')] : [])] : ['-an']),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-t', num(total),
    ...container,
    input.output,
  ];
  return { mode: 'reencode', args, totalSeconds: total, warnings };
}

export interface RunningFfmpeg {
  done: Promise<void>;
  cancel: () => void;
}

/** Runs ffmpeg with `-progress pipe:1` output, reporting how far through `totalSeconds` it is. */
export function runFfmpeg(ff: FfmpegPaths, args: string[], totalSeconds: number, onProgress: (fraction: number) => void): RunningFfmpeg {
  const child: ChildProcess = spawn(ff.ffmpeg, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let cancelled = false;
  let stderr = '';
  let buffered = '';
  child.stdout?.on('data', (chunk: Buffer) => {
    buffered += chunk.toString('utf-8');
    const lines = buffered.split('\n');
    buffered = lines.pop() ?? '';
    for (const line of lines) {
      const m = /^out_time_(?:us|ms)=(\d+)/.exec(line.trim());
      if (m && totalSeconds > 0) onProgress(Math.min(1, Number(m[1]) / 1e6 / totalSeconds));
    }
  });
  child.stderr?.on('data', (chunk: Buffer) => {
    stderr = (stderr + chunk.toString('utf-8')).slice(-4000);
  });
  const done = new Promise<void>((resolve, reject) => {
    child.on('error', (err) => reject(new Error(`Could not start ffmpeg: ${err.message}`)));
    child.on('close', (code) => {
      if (cancelled) reject(new Error('Assembly cancelled.'));
      else if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (exit ${code}): ${stderr.trim().split('\n').slice(-4).join(' ')}`));
    });
  });
  return {
    done,
    cancel: () => {
      cancelled = true;
      child.kill('SIGKILL');
    },
  };
}
