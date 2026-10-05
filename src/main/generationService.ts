import { DatabaseSync } from 'node:sqlite';
import * as fs from 'fs';
import * as path from 'path';
import { FAMILY_KIND } from '../shared/types';
import { generate as comfyGenerate } from './comfyui';
import { getHiddenWords, getImagesDir, getSourcesDir, getVideosDir } from './dbLocation';
import { keepSourceImage } from './sourceImages';
import { compileHiddenMatcher } from '../shared/hiddenWords';
import { insertGeneration } from './db';
import { insertTiming } from './timingStats';
import { faststartMp4 } from './mp4Faststart';
import { JobRunner } from './jobQueue';

// The model family of the last generation that finished. ComfyUI keeps a family's models loaded
// until something else needs the memory, so the next run of the same family starts "warm".
let lastRunFamily: string | null = null;

export function getLastRunFamily(): string | null {
  return lastRunFamily;
}

/**
 * The runner behind the job queue: sends one job to ComfyUI, saves the output file, and records
 * the generation and its timing. This is what used to be the body of the `generate` IPC handler,
 * moved here so every entry point (the UI today, outside clients later) runs identical work.
 */
export function createGenerationRunner(getDb: () => DatabaseSync | null): JobRunner {
  return async (job, { estimate, onProgress }) => {
    const db = getDb();
    if (!db) throw new Error('Database not initialized');
    const { family, params } = job;

    const warm = lastRunFamily === family;
    const output = await comfyGenerate(family, params, onProgress);
    lastRunFamily = family;

    // Images and videos are saved to separate folders under KVGenius_Data/output.
    const outputDir = FAMILY_KIND[family] === 'video' ? getVideosDir() : getImagesDir();
    fs.mkdirSync(outputDir, { recursive: true });
    const filename = `${Date.now()}-${params.seed}${output.extension}`;
    const imagePath = path.join(outputDir, filename);
    // ComfyUI's MP4s keep their index at the end of the file, which the in-app player can't
    // handle when served through kvimage:// - store them with the index up front instead.
    let bytes = output.bytes;
    if (output.extension.toLowerCase() === '.mp4') {
      try {
        bytes = faststartMp4(bytes) ?? bytes;
      } catch {
        // Keep the original bytes; playback may fail but the generation is still saved.
      }
    }
    fs.writeFileSync(imagePath, bytes);

    // How long it took vs what was predicted goes in its own table (timing_stats), holding only
    // timings and settings - never the prompt or image - so it outlives the generation.
    const t = output.timings;
    const timingId = insertTiming(db, {
      family,
      kind: FAMILY_KIND[family] === 'video' ? 'video' : 'image',
      width: params.width,
      height: params.height,
      steps: params.steps,
      cfg: params.cfg,
      length: params.length ?? null,
      warm,
      estimateMs: estimate?.totalMs ?? null,
      estimateGenerateMs: estimate?.generateMs ?? null,
      actualMs: t.totalMs,
      loadMs: t.loadMs,
      generateMs: t.loadMs === null ? t.totalMs : t.totalMs - t.loadMs,
      samplingMs: t.samplingMs,
      finishMs: t.finishMs,
      samplerSteps: t.samplerSteps,
      paceMs: t.paceMs,
    });
    // A prompt containing one of the user's hidden words is kept out of the Library (Settings > Hidden Content).
    const hidden = compileHiddenMatcher(getHiddenWords())(params.prompt);
    // A video keeps a copy of the image it was made from, so Re-rack can run it again in place.
    const keptSource =
      family === 'wan22-i2v' && params.sourceImagePath ? keepSourceImage(params.sourceImagePath, getSourcesDir()) : null;
    const record = insertGeneration(db, params, family, imagePath, timingId, hidden, keptSource);
    return { generationId: record.id };
  };
}
