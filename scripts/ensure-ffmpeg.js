// Makes sure the ffmpeg binary is on disk before the app is built and packaged.
//
// ffmpeg-static downloads its binary in an npm postinstall step. The shared release workflow
// (KVG_Standards release-electron.yml) installs with --ignore-scripts and then runs only the
// Electron and esbuild installers by hand, so without this the packaged app ships with no ffmpeg
// (ffprobe-static is fine: its binaries come inside the npm package). Runs as `prebuild`.
//
// The macOS build deliberately does not bundle ffmpeg (see mediaTools.ts), so it is skipped there.

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

if (process.platform === 'darwin') {
  console.log('ensure-ffmpeg: macOS build does not bundle ffmpeg, skipping.');
  process.exit(0);
}

function ffmpegPath() {
  try {
    return require('ffmpeg-static');
  } catch {
    return null;
  }
}

let binary = ffmpegPath();
if (!binary) {
  console.error('ensure-ffmpeg: ffmpeg-static is not installed (it is an optional dependency). Run `npm install`.');
  process.exit(1);
}

if (!fs.existsSync(binary)) {
  console.log('ensure-ffmpeg: downloading ffmpeg...');
  const installer = path.join(path.dirname(require.resolve('ffmpeg-static/package.json')), 'install.js');
  const result = spawnSync(process.execPath, [installer], { stdio: 'inherit' });
  if (result.status !== 0) {
    console.error('ensure-ffmpeg: the ffmpeg download failed.');
    process.exit(1);
  }
}

if (!fs.existsSync(binary)) {
  console.error(`ensure-ffmpeg: ffmpeg is still missing at ${binary}.`);
  process.exit(1);
}
console.log(`ensure-ffmpeg: ffmpeg is present (${binary}).`);
