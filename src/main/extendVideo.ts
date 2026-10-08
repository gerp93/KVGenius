import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import { getGenerationById } from './db';
import { FfmpegPaths, planJoin, probeMedia, runFfmpeg } from './mediaTools';

/** Frame rate Wan videos are made at; used only if the earlier video's own rate cannot be read. */
const FALLBACK_FPS = 16;

export interface JoinResult {
  /** How many frames the earlier video had - what the result starts with before the new clip. */
  extendedFrames: number;
}

/**
 * Joins the freshly made clip `clipPath` onto the end of the existing video `originalId` and writes the one longer video to `outputPath`.
 * The clip was made from the earlier video's last frame, so it is brought to the earlier video's size and frame rate (see `planJoin`). Returns
 * null when it cannot be done - ffmpeg missing, the earlier video gone or unreadable, ffmpeg failing - and leaves nothing at `outputPath`, so the
 * caller keeps the clip as it is rather than losing an expensive render over a joining problem.
 */
export async function joinOntoVideo(
  db: DatabaseSync,
  ff: FfmpegPaths | null,
  originalId: number,
  clipPath: string,
  outputPath: string
): Promise<JoinResult | null> {
  const partial = `${outputPath}.part`;
  try {
    if (!ff) return null;
    const original = getGenerationById(db, originalId);
    if (!original || !fs.existsSync(original.imagePath)) return null;
    const info = await probeMedia(ff, original.imagePath);
    if (!info.hasVideo || !info.width || !info.height) return null;
    const fps = info.fps ?? FALLBACK_FPS;
    await runFfmpeg(
      ff,
      planJoin({ first: original.imagePath, second: clipPath, output: partial, width: info.width, height: info.height, fps }),
      0,
      () => {}
    ).done;
    fs.renameSync(partial, outputPath);
    // Its own clip, plus whatever it was itself extended from; a video of unknown length falls back on its duration.
    const earlierFrames = (original.extendedFrames ?? 0) + (original.length ?? Math.round((info.durationSeconds ?? 0) * fps));
    return { extendedFrames: earlierFrames };
  } catch (err) {
    console.warn('Joining the extension onto the earlier video failed:', err instanceof Error ? err.message : err);
    return null;
  } finally {
    fs.rmSync(partial, { force: true });
  }
}
