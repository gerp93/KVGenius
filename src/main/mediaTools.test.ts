import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AssembleInput, planAssemble } from './mediaTools';

const base = (over: Partial<AssembleInput> = {}): AssembleInput => ({
  clips: [
    { path: '/c/a.mp4', durationSeconds: 5 },
    { path: '/c/b.mp4', durationSeconds: 5 },
    { path: '/c/c.mp4', durationSeconds: 5 },
  ],
  width: 640,
  height: 368,
  fps: 16,
  audio: { path: '/c/song.wav', durationSeconds: 20 },
  transition: 'cut',
  crossfadeSeconds: 0.5,
  end: 'trim_to_video',
  fadeSeconds: 2,
  output: '/o/out.mp4',
  listFilePath: '/o/list.txt',
  ...over,
});

test('plain cuts with the default end use stream copy and a concat list', () => {
  const plan = planAssemble(base());
  assert.equal(plan.mode, 'copy');
  assert.equal(plan.totalSeconds, 15);
  assert.equal(plan.listFileContent, "file '/c/a.mp4'\nfile '/c/b.mp4'\nfile '/c/c.mp4'\n");
  assert.ok(plan.args.includes('copy'));
  assert.equal(plan.args[plan.args.indexOf('-t') + 1], '15');
  assert.equal(plan.args[plan.args.length - 1], '/o/out.mp4');
});

test('quotes in file names are escaped for the concat list', () => {
  const plan = planAssemble(base({ clips: [{ path: "/c/it's.mp4", durationSeconds: 3 }] }));
  assert.equal(plan.listFileContent, "file '/c/it'\\''s.mp4'\n");
});

test('a trimmed clip forces a re-encode and shortens the total', () => {
  const plan = planAssemble(base({ clips: [{ path: '/c/a.mp4', durationSeconds: 5, trimSeconds: 2 }, { path: '/c/b.mp4', durationSeconds: 5 }] }));
  assert.equal(plan.mode, 'reencode');
  assert.equal(plan.totalSeconds, 7);
  const graph = plan.args[plan.args.indexOf('-filter_complex') + 1];
  assert.match(graph, /\[0:v\]trim=duration=2,/);
  assert.match(graph, /concat=n=2:v=1:a=0\[vcat\]/);
});

test('a trim longer than the clip is ignored', () => {
  const plan = planAssemble(base({ clips: [{ path: '/c/a.mp4', durationSeconds: 5, trimSeconds: 9 }] }));
  assert.equal(plan.mode, 'copy');
  assert.equal(plan.totalSeconds, 5);
});

test('crossfades chain xfade with offsets and shorten the total by the overlap', () => {
  const plan = planAssemble(base({ transition: 'crossfade', crossfadeSeconds: 1 }));
  assert.equal(plan.mode, 'reencode');
  assert.equal(plan.totalSeconds, 13);
  const graph = plan.args[plan.args.indexOf('-filter_complex') + 1];
  assert.match(graph, /\[v0\]\[v1\]xfade=transition=fade:duration=1:offset=4\[x1\]/);
  assert.match(graph, /\[x1\]\[v2\]xfade=transition=fade:duration=1:offset=8\[vcat\]/);
});

test('an over-long crossfade is shortened with a warning', () => {
  const plan = planAssemble(base({ clips: [{ path: '/a', durationSeconds: 1 }, { path: '/b', durationSeconds: 1 }], transition: 'crossfade', crossfadeSeconds: 5 }));
  assert.ok(plan.warnings.some((w) => /Crossfade shortened/.test(w)));
  assert.ok(plan.totalSeconds > 1);
});

test('trim_to_audio holds the last frame when the clips are shorter than the track', () => {
  const plan = planAssemble(base({ end: 'trim_to_audio' }));
  assert.equal(plan.mode, 'reencode');
  assert.equal(plan.totalSeconds, 20);
  const graph = plan.args[plan.args.indexOf('-filter_complex') + 1];
  assert.match(graph, /tpad=stop_mode=clone:stop_duration=5/);
  assert.ok(plan.warnings.some((w) => /last frame is held/.test(w)));
});

test('trim_to_audio cuts the video when the clips outlast the track', () => {
  const plan = planAssemble(base({ end: 'trim_to_audio', audio: { path: '/c/s.wav', durationSeconds: 12 } }));
  assert.equal(plan.totalSeconds, 12);
  assert.equal(plan.args[plan.args.indexOf('-t') + 1], '12');
  assert.ok(!plan.args.join(' ').includes('tpad'));
});

test('fade_out fades video and audio at the end', () => {
  const plan = planAssemble(base({ end: 'fade_out', fadeSeconds: 2 }));
  const graph = plan.args[plan.args.indexOf('-filter_complex') + 1];
  assert.match(graph, /fade=t=out:st=13:d=2/);
  assert.equal(plan.args[plan.args.indexOf('-af') + 1], 'afade=t=out:st=13:d=2');
});

test('audio shorter than the video is padded with silence and warned about', () => {
  const plan = planAssemble(base({ audio: { path: '/c/s.wav', durationSeconds: 4 } }));
  assert.equal(plan.args[plan.args.indexOf('-af') + 1], 'apad');
  assert.ok(plan.warnings.some((w) => /shorter than the video/.test(w)));
});

test('audio longer than the video is cut and warned about', () => {
  const plan = planAssemble(base());
  assert.ok(plan.warnings.some((w) => /is cut at 15s/.test(w)));
});

test('without audio the output has no audio stream', () => {
  const plan = planAssemble(base({ audio: null }));
  assert.ok(plan.args.includes('-an'));
});

test('no clips is an error', () => {
  assert.throws(() => planAssemble(base({ clips: [] })), /At least one clip/);
});
