// wan22-i2v renders at 16fps (CreateVideo node in its template) and needs a frame count of
// 4n+1, so a duration in seconds snaps to quarter-seconds (81 frames = 5s).
export const VIDEO_FPS = 16;

export function secondsToFrames(seconds: number): number {
  const clamped = Math.min(12, Math.max(1, seconds || 0));
  return 4 * Math.round(clamped * (VIDEO_FPS / 4)) + 1;
}

export function framesToSeconds(frames: number): number {
  return Math.round(((frames - 1) / VIDEO_FPS) * 4) / 4;
}
