import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { initDatabase, insertGeneration } from './db';
import { joinOntoVideo } from './extendVideo';
import { findFfmpeg, probeMedia } from './mediaTools';

const ff = findFfmpeg();
const params = { prompt: 'a fox runs', width: 64, height: 96, seed: 3, steps: 4, cfg: 1, length: 32 };

function makeClip(file: string, color: string, frames: number, size: string): void {
  execFileSync(ff!.ffmpeg, ['-y', '-v', 'error', '-f', 'lavfi', '-i', `color=c=${color}:s=${size}:r=16`, '-frames:v', String(frames), '-pix_fmt', 'yuv420p', file]);
}

const frameCount = (file: string) =>
  Number(execFileSync(ff!.ffprobe, ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', file]).toString().trim());

test('a clip is joined onto the end of the video it continues, and the frames of the earlier part are reported', { skip: ff ? false : 'ffmpeg is not installed' }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-extend-'));
  try {
    const db = initDatabase(':memory:');
    const original = path.join(dir, 'original.mp4');
    makeClip(original, 'red', 32, '64x96');
    const record = insertGeneration(db, params, 'wan22-i2v', original);

    const clip = path.join(dir, 'clip.mp4');
    makeClip(clip, 'blue', 17, '32x48'); // a different size: it is brought to the original's
    const out = path.join(dir, 'joined.mp4');
    const joined = await joinOntoVideo(db, ff, record.id, clip, out);
    assert.deepEqual(joined, { extendedFrames: 32 });
    const info = await probeMedia(ff!, out);
    assert.deepEqual([info.width, info.height], [64, 96]);
    assert.equal(frameCount(out), 48, '32 + 17 frames, less the repeated first frame of the clip');

    // extending the extended video again counts everything before the new clip
    const second = insertGeneration(db, { ...params, length: 17 }, 'wan22-i2v', out, null, false, null, null, 32);
    const again = await joinOntoVideo(db, ff, second.id, clip, path.join(dir, 'joined2.mp4'));
    assert.deepEqual(again, { extendedFrames: 32 + 17 });
    assert.equal(second.extendedFrames, 32);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('when the join cannot be done nothing is left behind, so the caller keeps the clip as it is', { skip: ff ? false : 'ffmpeg is not installed' }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-extend-'));
  try {
    const db = initDatabase(':memory:');
    const clip = path.join(dir, 'clip.mp4');
    makeClip(clip, 'blue', 17, '64x96');
    const out = path.join(dir, 'joined.mp4');
    assert.equal(await joinOntoVideo(db, ff, 999, clip, out), null, 'no such video');
    const gone = insertGeneration(db, params, 'wan22-i2v', path.join(dir, 'missing.mp4'));
    assert.equal(await joinOntoVideo(db, ff, gone.id, clip, out), null, 'its file is gone');
    assert.equal(await joinOntoVideo(db, null, gone.id, clip, out), null, 'no ffmpeg');
    assert.deepEqual(fs.readdirSync(dir), ['clip.mp4']);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
