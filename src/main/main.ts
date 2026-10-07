import { app, BrowserWindow, clipboard, ClipboardItem, ipcMain, dialog, nativeImage, protocol, shell } from 'electron';
import { DatabaseSync } from 'node:sqlite';
import * as path from 'path';
import * as fs from 'fs';
import { autoUpdater } from 'electron-updater';

import { setupApplicationMenu, attachContextMenu } from './menu';
import {
  getEffectiveDbPath,
  getDefaultDbPath,
  isUsingDefaultDbLocation,
  setDbPath,
  resetToDefaultDbPath,
  revealDbInFileManager,
  getImagesDir,
  getVideosDir,
  getGifsDir,
  getSourcesDir,
  getLegacyOutputDir,
  dbPathInsideFolder,
  enforceDevDatabaseIsolation,
  migrateLegacyDefaultDbLocation,
  getEffectiveComfyUIHost,
  setComfyUIHost,
  resetComfyUIHost,
  getEffectiveTheme,
  setTheme,
  getPromptSlots,
  getActivePromptSlotId,
  savePromptSlots,
  getApiEnabled,
  setApiEnabled,
  getApiToken,
  getFfmpegOverride,
  setFfmpegOverride,
  getComfyUILaunchPath,
  setComfyUILaunchPath,
  getHiddenWords,
  setHiddenWords,
  getTrashDir,
  getCleanupSettings,
  saveCleanupSettings,
} from './dbLocation';
import { compileHiddenMatcher } from '../shared/hiddenWords';
import { normalizeDays, updateCleanupSettings } from '../shared/cleanup';
import { PromptSlot } from '../shared/promptSlots';
import {
  initDatabase,
  insertGeneration,
  getGenerationById,
  listGenerations,
  countGenerations,
  listGenerationRefs,
  listImageExtensions,
  setGenerationHidden,
  applyHiddenRule,
  moveLegacyOutput,
  setGenerationPinned,
  countPinnedWithPrompt,
  findDuplicateGeneration,
  listPinnedGenerations,
} from './db';
import {
  isAvailable as comfyIsAvailable,
  cancelCurrentGeneration,
  listUpscaleModels,
  GenerationCancelledError,
  DEFAULT_COMFYUI_HOST,
} from './comfyui';
import { JobQueue } from './jobQueue';
import { createGenerationRunner, getLastRunFamily } from './generationService';
import { AssemblyManager } from './assembly';
import { ApiService } from './apiService';
import { LocalApi, removeDiscoveryFile, startLocalApi, writeDiscoveryFile } from './localApi';
import { FfmpegPaths, findFfmpeg, planGif, probeMedia, runFfmpeg } from './mediaTools';
import { GIF_FAMILY } from '../shared/gif';
import { detectComfyUIProgram, launchComfyUIProgram } from './comfyLauncher';
import { ComfyUILauncherInfo, ComfyUILaunchResult, FAMILY_KIND, GenerationKind, GenerationParams, LibraryListOptions, McpInfo } from '../shared/types';
import { estimateRun } from '../shared/estimator';
import { clearTimingStats, insertTiming, listTimingRows } from './timingStats';
import { isHardpointReachable, openHardpoint } from './hardpointLaunch';
import { MEDIA_SCHEME, MEDIA_SCHEME_PRIVILEGES, VIDEO_EXTENSIONS, handleMediaRequest, isAllowedMediaPath } from './mediaProtocol';
import { MediaServer, startMediaServer } from './mediaServer';
import { applyFavorite, syncFavoriteFiles } from './favorites';
import {
  deleteFromTrash,
  emptyTrash,
  listTrashed,
  moveToTrash,
  previewCleanup,
  restoreFromTrash,
  runCleanup,
  trashStats,
  Recycle,
} from './trash';
import { startCleanupSchedule } from './cleanupScheduler';
import { uniqueNames, writeZip } from './zipWriter';
import { diagnoseVideo } from './videoDiagnostics';

// Dev and packaged builds must never share a userData/appData folder, or
// enforceDevDatabaseIsolation() below can never tell them apart (it compares
// against exactly this path) - it would either never fire (silently useless)
// or, as found by actually running this in dev, always fire (dev's default
// path IS the packaged path without this line), popping a blocking dialog
// on every single dev launch. Must be set before any app.getPath() call.
app.setName(app.isPackaged ? 'kvgenius' : 'kvgenius-dev');

// Generated images load through this instead of a raw `file://` src - Electron refuses to load
// `file://` subresources from a page whose own origin isn't `file:`, which is true of the dev
// window (loads Vite's `http://localhost:5173`). A custom scheme carries no such restriction
// and behaves identically in dev and packaged builds. Must be registered before app is ready.
protocol.registerSchemesAsPrivileged([
  { scheme: MEDIA_SCHEME, privileges: MEDIA_SCHEME_PRIVILEGES },
]);

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

function logStartupFailure(context: string, error: unknown): void {
  const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
  const line = `[${new Date().toISOString()}] ${context}: ${message}\n`;
  process.stderr.write(line);
  try {
    fs.mkdirSync(app.getPath('userData'), { recursive: true });
    fs.appendFileSync(path.join(app.getPath('userData'), 'main.log'), line);
  } catch {
    // Log write failing (e.g. the same locked-folder problem that caused the original error)
    // must not stop the dialog below from at least trying to show.
  }
}

function showStartupFailureDialog(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  dialog.showErrorBox(
    'KVGenius failed to start',
    `${message}\n\nDetails were written to:\n${path.join(app.getPath('userData'), 'main.log')}`
  );
}

process.on('uncaughtException', (error) => {
  logStartupFailure('uncaughtException', error);
  showStartupFailureDialog(error);
  app.quit();
});

let mainWindow: BrowserWindow | null = null;
let db: DatabaseSync | null = null;
let jobQueue: JobQueue | null = null;
let assemblies: AssemblyManager | null = null;
let apiService: ApiService | null = null;
let localApi: LocalApi | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    icon: path.join(__dirname, '..', '..', 'build', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, '..', 'preload', 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  attachContextMenu(mainWindow);

  if (!app.isPackaged) {
    void mainWindow.loadURL('http://localhost:5173');
  } else {
    // __dirname here is dist/main/main (main.ts -> src/main/main.ts, outDir dist/main,
    // rootDir src) while the renderer build lands at dist/renderer (vite.config.ts's
    // build.outDir) - a sibling of dist/main, not a child of it. One ".." only reaches
    // dist/main; confirmed by actually running the packaged AppImage, which failed to
    // load the window with ERR_FILE_NOT_FOUND before this fix.
    void mainWindow.loadFile(path.join(__dirname, '..', '..', 'renderer', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Source images picked with the native file dialog live outside the images directory; each one
// is allowed individually so the Generate page can preview it.
const pickedSourceImages = new Set<string>();

function outputDirs() {
  return { images: getImagesDir(), videos: getVideosDir(), gifs: getGifsDir(), legacy: getLegacyOutputDir() };
}

function videoFamilyList(): string[] {
  return Object.keys(FAMILY_KIND).filter((family) => FAMILY_KIND[family] === 'video');
}

/** Every folder the renderer may be shown files from: the output folders, the kept source images, and the Trash. */
function mediaDirs(): string[] {
  return [getImagesDir(), getVideosDir(), getGifsDir(), getSourcesDir(), getTrashDir(), getLegacyOutputDir()];
}

/** Sends a file to the operating system's Recycle Bin / Trash. */
const recycleFile: Recycle = (file) => shell.trashItem(path.resolve(file));

/** Library list options from the renderer, with anything unexpected dropped. */
function cleanListOptions(options: LibraryListOptions | undefined): LibraryListOptions {
  return { grouped: options?.grouped === true, prompt: typeof options?.prompt === 'string' ? options.prompt : null };
}

/** Ids from the renderer, keeping only whole numbers. */
function cleanIds(ids: unknown): number[] {
  return Array.isArray(ids) ? ids.filter((id): id is number => Number.isInteger(id)) : [];
}

function registerImageProtocol(): void {
  protocol.handle(MEDIA_SCHEME, (request) => handleMediaRequest(request, mediaDirs(), pickedSourceImages));
}

let mediaServer: MediaServer | null = null;

/** Images load through the kvimage:// protocol; videos through the local HTTP media server (see
 * mediaServer.ts for why). The preload script builds URLs with the same rule. */
function imageUrlFor(imagePath: string): string {
  if (mediaServer && VIDEO_EXTENSIONS.includes(path.extname(imagePath).toLowerCase())) {
    return `${mediaServer.base}/${encodeURIComponent(imagePath)}`;
  }
  return `kvimage://${encodeURIComponent(imagePath)}`;
}

// ffmpeg is looked up by spawning it, so remember the answer. A miss is only remembered briefly, so
// installing ffmpeg (or fixing the path in Settings) is noticed without restarting the app.
let ffmpegCache: { override: string | null; at: number; paths: FfmpegPaths | null } | null = null;

function getFfmpeg(): FfmpegPaths | null {
  const override = getFfmpegOverride();
  const fresh =
    ffmpegCache && ffmpegCache.override === override && (ffmpegCache.paths !== null || Date.now() - ffmpegCache.at < 5000);
  if (!fresh) ffmpegCache = { override, at: Date.now(), paths: findFfmpeg(override) };
  return ffmpegCache?.paths ?? null;
}

function comfyLauncherInfo(): ComfyUILauncherInfo {
  return { configured: getComfyUILaunchPath(), detected: detectComfyUIProgram() };
}

/** Asks for the program that starts ComfyUI and remembers it. Null if the dialog is dismissed. */
async function chooseComfyUIProgram(): Promise<string | null> {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose the program that starts ComfyUI',
    message: 'ComfyUI Desktop, or a run script / AppImage for the standalone version.',
    properties: ['openFile'],
    ...(process.platform === 'win32' ? { filters: [{ name: 'Programs and scripts', extensions: ['exe', 'bat', 'cmd'] }] } : {}),
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  setComfyUILaunchPath(result.filePaths[0]);
  return result.filePaths[0];
}

async function launchComfyUI(): Promise<ComfyUILaunchResult> {
  if (await comfyIsAvailable()) return { status: 'already-running' };
  const target = getComfyUILaunchPath() ?? detectComfyUIProgram() ?? (await chooseComfyUIProgram());
  if (!target) return { status: 'cancelled' };
  const result = await launchComfyUIProgram(target, (appPath) => shell.openPath(appPath));
  return result.status === 'launched' ? { status: 'launched' } : result;
}

function discoveryFilePath(): string {
  return path.join(app.getPath('userData'), 'mcp-api.json');
}

/** Turns the local control API on: the tool service, the HTTP server, and the discovery file the
 * MCP stdio shim reads to find it. No-op if it is already running. */
async function startApi(): Promise<void> {
  if (localApi || !db || !jobQueue) return;
  if (!assemblies) {
    assemblies = new AssemblyManager(db, {
      ffmpeg: getFfmpeg,
      outputDir: getVideosDir,
      tempDir: () => path.join(app.getPath('temp'), 'kvgenius-assembly'),
    });
  }
  if (!apiService) {
    apiService = new ApiService({
      db,
      queue: jobQueue,
      assemblies,
      ffmpeg: getFfmpeg,
      comfyAvailable: comfyIsAvailable,
      imageSize: (file) => {
        const size = nativeImage.createFromPath(file).getSize();
        return size.width > 0 && size.height > 0 ? size : null;
      },
      imagePreviewFallback: async (file) => {
        const image = nativeImage.createFromPath(file);
        if (image.isEmpty()) return null;
        const { width } = image.getSize();
        return (width > 512 ? image.resize({ width: 512 }) : image).toJPEG(80);
      },
    });
  }
  const service = apiService;
  const token = getApiToken();
  localApi = await startLocalApi({
    token,
    preferredPort: 47615,
    version: app.getVersion(),
    handler: (tool, args) => service.callTool(tool, args),
  });
  writeDiscoveryFile(discoveryFilePath(), { port: localApi.port, token, pid: process.pid, version: app.getVersion() });
}

async function stopApi(): Promise<void> {
  const api = localApi;
  localApi = null;
  removeDiscoveryFile(discoveryFilePath());
  await api?.close();
}

function mcpInfo(): McpInfo {
  const ff = getFfmpeg();
  const snippet = {
    mcpServers: {
      kvgenius: {
        command: process.execPath,
        args: [path.join(__dirname, '..', 'mcp', 'shim.js')],
        env: {
          ELECTRON_RUN_AS_NODE: '1',
          KVGENIUS_API_FILE: discoveryFilePath(),
          KVGENIUS_VERSION: app.getVersion(),
        },
      },
    },
  };
  return {
    enabled: getApiEnabled(),
    running: localApi !== null,
    port: localApi?.port ?? null,
    configSnippet: JSON.stringify(snippet, null, 2),
    ffmpeg: { available: ff !== null, path: ff?.ffmpeg ?? null, override: getFfmpegOverride() },
  };
}

function registerIpcHandlers(): void {
  ipcMain.handle(
    'generate',
    async (
      _event,
      family: string,
      params: GenerationParams,
      estimate: { totalMs: number | null; generateMs: number | null } | null
    ) => {
    if (!db || !jobQueue) throw new Error('Database not initialized');

    // The Generate page keeps its own list of what it has queued and awaits one job at a time, so
    // this waits for the job to finish; the main-process queue is what puts it in line with
    // anything else (e.g. an outside client) that is also using the GPU.
    const submitted = jobQueue.submit({ family, params, estimate, source: 'ui' });
    const job = await jobQueue.wait(submitted.id);
    if (job.status === 'cancelled') throw new GenerationCancelledError('Generation cancelled.');
    const record = job.generationId === null ? null : getGenerationById(db, job.generationId);
    if (job.status !== 'done' || !record) throw new Error(job.error ?? 'Generation failed.');
    return { record, imageUrl: imageUrlFor(record.imagePath) };
    }
  );

  ipcMain.handle(
    'estimateGeneration',
    (_event, family: string, params: GenerationParams, previousFamily?: string | null) => {
      if (!db) return null;
      const before = previousFamily === undefined ? getLastRunFamily() : previousFamily;
      return estimateRun(listTimingRows(db, 200), {
        family,
        kind: FAMILY_KIND[family] === 'video' ? 'video' : 'image',
        width: params.width,
        height: params.height,
        steps: params.steps,
        cfg: params.cfg,
        lengthFrames: params.length ?? null,
        warm: before === family,
      });
    }
  );

  ipcMain.handle('getTimingStats', () => {
    if (!db) throw new Error('Database not initialized');
    return listTimingRows(db);
  });

  ipcMain.handle('clearTimingStats', () => {
    if (!db) throw new Error('Database not initialized');
    clearTimingStats(db);
  });

  ipcMain.handle('cancelGeneration', async () => {
    await jobQueue?.cancelRunning();
  });

  const videoFamilies = videoFamilyList();

  ipcMain.handle(
    'listGenerations',
    (
      _event,
      kind: GenerationKind,
      limit: number,
      beforeId: number | null,
      favoritesOnly: boolean,
      showHidden: boolean,
      extension?: string | null,
      options?: LibraryListOptions
    ) => {
      if (!db) throw new Error('Database not initialized');
      const safeLimit = Math.min(Math.max(Math.floor(limit) || 0, 1), 200);
      return listGenerations(
        db,
        videoFamilies,
        kind === 'video' ? 'video' : 'image',
        safeLimit,
        beforeId,
        !!favoritesOnly,
        !!showHidden,
        extension ?? null,
        cleanListOptions(options)
      );
    }
  );

  ipcMain.handle('countGenerations', (_event, favoritesOnly: boolean, showHidden: boolean, imageExtension?: string | null, options?: LibraryListOptions) => {
    if (!db) throw new Error('Database not initialized');
    return countGenerations(db, videoFamilies, !!favoritesOnly, !!showHidden, imageExtension ?? null, cleanListOptions(options));
  });

  ipcMain.handle('listImageExtensions', () => {
    if (!db) throw new Error('Database not initialized');
    return listImageExtensions(db, videoFamilies);
  });

  ipcMain.handle(
    'listGenerationRefs',
    (_event, kind: GenerationKind, favoritesOnly: boolean, showHidden: boolean, extension?: string | null, options?: LibraryListOptions) => {
      if (!db) throw new Error('Database not initialized');
      return listGenerationRefs(db, videoFamilies, kind === 'video' ? 'video' : 'image', !!favoritesOnly, !!showHidden, extension ?? null, cleanListOptions(options));
    }
  );

  ipcMain.handle('setGenerationHidden', (_event, id: number, hidden: boolean) => {
    if (!db) throw new Error('Database not initialized');
    setGenerationHidden(db, id, !!hidden);
  });

  ipcMain.handle('listUpscaleModels', () => listUpscaleModels());

  ipcMain.handle('convertToGif', async (_event, id: number, options: { fps: number; width: number }) => {
    if (!db) throw new Error('Database not initialized');
    const ff = getFfmpeg();
    if (!ff) throw new Error('ffmpeg was not found. Install it or choose it in Settings.');
    const source = getGenerationById(db, id);
    if (!source || FAMILY_KIND[source.modelFamily] !== 'video') throw new Error('That item is not a video.');
    const fps = Math.min(Math.max(Math.round(options.fps) || 15, 1), 30);
    const width = Math.min(Math.max(Math.round(options.width) || 480, 64), 4096);

    const info = await probeMedia(ff, source.imagePath);
    if (!info.hasVideo || !info.width || !info.height) throw new Error('Could not read the video.');
    const gifWidth = Math.min(width, info.width);
    const gifHeight = Math.max(2, Math.round((info.height * gifWidth) / info.width / 2) * 2);

    const outputDir = getGifsDir();
    fs.mkdirSync(outputDir, { recursive: true });
    const finalPath = path.join(outputDir, `${Date.now()}-${source.seed}.gif`);
    const partial = `${finalPath}.part`;
    try {
      await runFfmpeg(ff, planGif({ input: source.imagePath, output: partial, fps, width }), info.durationSeconds ?? 0, () => {}).done;
      fs.renameSync(partial, finalPath);
    } finally {
      fs.rmSync(partial, { force: true });
    }

    const hidden = compileHiddenMatcher(getHiddenWords())(source.prompt);
    const record = insertGeneration(
      db,
      { prompt: source.prompt, width: gifWidth, height: gifHeight, seed: source.seed, steps: source.steps, cfg: source.cfg },
      GIF_FAMILY,
      finalPath,
      null,
      hidden
    );
    return { record, imageUrl: imageUrlFor(record.imagePath) };
  });

  ipcMain.handle('getHiddenWords', () => getHiddenWords());
  ipcMain.handle('setHiddenWords', (_event, words: string[]) => setHiddenWords(Array.isArray(words) ? words : []));
  ipcMain.handle('applyHiddenWords', () => {
    if (!db) throw new Error('Database not initialized');
    return applyHiddenRule(db, compileHiddenMatcher(getHiddenWords()));
  });

  ipcMain.handle('getFileSize', async (_event, imagePath: string) => {
    try {
      return (await fs.promises.stat(imagePath)).size;
    } catch {
      return null;
    }
  });

  ipcMain.handle('exportGenerations', async (_event, imagePaths: string[]) => {
    if (!mainWindow) return { status: 'cancelled' };
    const stamp = new Date().toISOString().slice(0, 10);
    const result = await dialog.showSaveDialog(mainWindow, {
      title: 'Export to zip',
      defaultPath: `KVGenius-export-${stamp}.zip`,
      filters: [{ name: 'Zip archive', extensions: ['zip'] }],
    });
    if (result.canceled || !result.filePath) return { status: 'cancelled' };

    const existing = imagePaths.filter((p) => fs.existsSync(p));
    const names = uniqueNames(existing.map((p) => path.basename(p)));
    await writeZip(
      result.filePath,
      existing.map((sourcePath, i) => ({ sourcePath, name: names[i] }))
    );
    return { status: 'saved', path: result.filePath, count: existing.length };
  });

  ipcMain.handle('setGenerationFavorite', (_event, id: number, favorite: boolean) => {
    if (!db) throw new Error('Database not initialized');
    return applyFavorite(db, id, !!favorite, outputDirs(), videoFamilies);
  });

  ipcMain.handle('imageUrlFor', (_event, imagePath: string) => imageUrlFor(imagePath));

  ipcMain.handle('chooseSourceImage', async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Choose a source image for video mode',
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] }],
      properties: ['openFile'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    pickedSourceImages.add(path.resolve(result.filePaths[0]));
    return result.filePaths[0];
  });

  // The Tools > Upscale picker: any number of images at once. Each is allowed to be shown and sent to ComfyUI.
  ipcMain.handle('chooseSourceImages', async () => {
    if (!mainWindow) return [];
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Choose images to upscale',
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] }],
      properties: ['openFile', 'multiSelections'],
    });
    if (result.canceled) return [];
    for (const file of result.filePaths) pickedSourceImages.add(path.resolve(file));
    return result.filePaths;
  });

  // Copies a picture to the clipboard so it can be pasted into other apps. Done here, from the file, so
  // it works for whatever the app can show - but only for files the app itself serves, never any path.
  ipcMain.handle('copyImageToClipboard', async (_event, imagePath: string) => {
    const resolved = path.resolve(String(imagePath));
    if (!isAllowedMediaPath(resolved, mediaDirs(), pickedSourceImages)) throw new Error('That file is not one the app can copy.');
    const image = nativeImage.createFromPath(resolved);
    if (image.isEmpty()) throw new Error('Could not read that file as an image (a video cannot be copied as a picture).');
    // As a PNG whatever the file is, so any app that takes a pasted picture can use it.
    await clipboard.write([new ClipboardItem({ 'image/png': new Blob([image.toPNG()], { type: 'image/png' }) })]);
  });

  ipcMain.handle('revealGenerationInFileManager', (_event, imagePath: string) => {
    shell.showItemInFolder(imagePath);
  });

  ipcMain.handle('diagnoseVideo', (_event, imagePath: string) => diagnoseVideo(imagePath));

  ipcMain.handle('openGenerationExternally', async (_event, imagePath: string) => {
    const err = await shell.openPath(imagePath);
    if (err) throw new Error(err);
  });

  ipcMain.handle('saveGenerationAs', async (_event, imagePath: string) => {
    if (!mainWindow) return false;
    const result = await dialog.showSaveDialog(mainWindow, {
      title: 'Save As',
      defaultPath: path.basename(imagePath),
    });
    if (result.canceled || !result.filePath) return false;
    fs.copyFileSync(imagePath, result.filePath);
    return true;
  });

  ipcMain.handle('setGenerationPinned', (_event, id: number, pinned: boolean) => {
    if (!db) throw new Error('Database not initialized');
    setGenerationPinned(db, id, !!pinned);
    const record = getGenerationById(db, id);
    return { groupSize: pinned && record ? countPinnedWithPrompt(db, record.prompt) : 0 };
  });

  // Would generating these settings only repeat something already in the Library? (Generate's button asks.)
  ipcMain.handle('findDuplicateGeneration', (_event, family: string, params: GenerationParams) => {
    if (!db) throw new Error('Database not initialized');
    if (typeof family !== 'string' || !params || typeof params.prompt !== 'string') return null;
    return findDuplicateGeneration(db, family, {
      prompt: params.prompt,
      width: params.width,
      height: params.height,
      seed: params.seed,
      steps: params.steps,
      cfg: params.cfg,
      length: params.length ?? null,
      sourceImagePath: params.sourceImagePath ?? null,
    });
  });

  ipcMain.handle('listPinnedGenerations', (_event, showHidden: boolean) => {
    if (!db) throw new Error('Database not initialized');
    return listPinnedGenerations(db, !!showHidden);
  });

  // Library cleanup and the Trash.
  ipcMain.handle('getCleanupSettings', () => getCleanupSettings());

  ipcMain.handle('setCleanupSettings', (_event, patch: Parameters<typeof updateCleanupSettings>[1]) =>
    saveCleanupSettings(updateCleanupSettings(getCleanupSettings(), patch ?? {}, new Date()))
  );

  ipcMain.handle('previewCleanup', (_event, days: number) => {
    if (!db) throw new Error('Database not initialized');
    return previewCleanup(db, normalizeDays(days, 30));
  });

  ipcMain.handle('runCleanup', (_event, days: number) => {
    if (!db) throw new Error('Database not initialized');
    return runCleanup(db, normalizeDays(days, 30), getTrashDir());
  });

  // Deleting in the app moves to the Trash (no confirmation: it can be restored from there). Favorites and
  // pinned items are only moved when the caller says the user asked for that item specifically.
  ipcMain.handle('trashGenerations', (_event, ids: number[], options?: { includeKept?: boolean }) => {
    if (!db) throw new Error('Database not initialized');
    return moveToTrash(db, cleanIds(ids), getTrashDir(), { includeKept: options?.includeKept === true });
  });

  ipcMain.handle('getTrashStats', () => {
    if (!db) throw new Error('Database not initialized');
    return trashStats(db);
  });

  ipcMain.handle('listTrashed', (_event, limit: number, beforeId: number | null) => {
    if (!db) throw new Error('Database not initialized');
    const safeLimit = Math.min(Math.max(Math.floor(limit) || 0, 1), 200);
    return listTrashed(db, safeLimit, Number.isInteger(beforeId) ? beforeId : null);
  });

  ipcMain.handle('restoreGenerations', (_event, ids: number[]) => {
    if (!db) throw new Error('Database not initialized');
    return restoreFromTrash(db, cleanIds(ids));
  });

  // Step two of a delete: the files go to the operating system's Recycle Bin.
  ipcMain.handle('deleteTrashed', (_event, ids: number[]) => {
    if (!db) throw new Error('Database not initialized');
    return deleteFromTrash(db, cleanIds(ids), getSourcesDir(), recycleFile);
  });

  ipcMain.handle('emptyTrash', () => {
    if (!db) throw new Error('Database not initialized');
    return emptyTrash(db, getSourcesDir(), recycleFile);
  });

  ipcMain.handle('getPromptSlots', () => ({
    slots: getPromptSlots(),
    activeId: getActivePromptSlotId(),
  }));

  ipcMain.handle('savePromptSlots', (_event, slots: PromptSlot[], activeId: string) => {
    savePromptSlots(slots, activeId);
  });

  ipcMain.handle('getComfyUIHost', () => ({
    host: getEffectiveComfyUIHost(),
    defaultHost: DEFAULT_COMFYUI_HOST,
  }));

  ipcMain.handle('setComfyUIHost', (_event, host: string) => {
    setComfyUIHost(host);
  });

  ipcMain.handle('resetComfyUIHost', () => {
    resetComfyUIHost();
  });

  ipcMain.handle('checkComfyUIConnection', () => comfyIsAvailable());

  ipcMain.handle('getTheme', () => getEffectiveTheme());

  ipcMain.handle('setTheme', (_event, themeId: string) => {
    setTheme(themeId);
  });

  ipcMain.handle('getDbInfo', () => {
    const dbPath = getEffectiveDbPath();
    let sizeBytes: number | null = null;
    try {
      sizeBytes = fs.statSync(dbPath).size;
    } catch {
      // Not created yet (e.g. a freshly chosen/reset location before first write).
    }
    return {
      path: dbPath,
      isDefault: isUsingDefaultDbLocation(),
      defaultPath: getDefaultDbPath(),
      sizeBytes,
    };
  });

  ipcMain.handle('revealDbInFileManager', () => revealDbInFileManager());

  // The database must be closed before its file is copied/adopted (setDbPath's job), and a
  // live node:sqlite connection can't just be repointed at a different path afterward - the
  // simplest correct fix is a full relaunch, which re-opens at whatever getEffectiveDbPath()
  // now resolves to. Matches the standard's own "then restart the app" requirement.
  function relocateAndRelaunch(newPath: string): void {
    jobQueue = null;
    db?.close();
    db = null;
    setDbPath(newPath);
    app.relaunch();
    app.exit();
  }

  ipcMain.handle('chooseExistingDb', async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Choose an existing KVGenius database',
      defaultPath: getEffectiveDbPath(),
      filters: [{ name: 'KVGenius database', extensions: ['db'] }],
      properties: ['openFile'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    relocateAndRelaunch(result.filePaths[0]);
    return result.filePaths[0];
  });

  ipcMain.handle('chooseNewDbLocation', async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Choose a parent folder for the KVGenius database',
      message: "A 'KVGenius_Data' folder will be created inside whatever you pick, holding the database and generated images together.",
      properties: ['openDirectory', 'createDirectory'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    const newPath = dbPathInsideFolder(result.filePaths[0]);
    relocateAndRelaunch(newPath);
    return newPath;
  });

  ipcMain.handle('resetDbToDefault', () => {
    jobQueue = null;
    db?.close();
    db = null;
    resetToDefaultDbPath();
    app.relaunch();
    app.exit();
  });

  ipcMain.handle('launchComfyUI', () => launchComfyUI());
  ipcMain.handle('getComfyUILauncher', () => comfyLauncherInfo());
  ipcMain.handle('chooseComfyUILauncher', async () => ((await chooseComfyUIProgram()) ? comfyLauncherInfo() : null));
  ipcMain.handle('clearComfyUILauncher', () => {
    setComfyUILaunchPath(null);
    return comfyLauncherInfo();
  });

  ipcMain.handle('getMcpInfo', () => mcpInfo());

  ipcMain.handle('setMcpEnabled', async (_event, enabled: boolean) => {
    setApiEnabled(!!enabled);
    if (enabled) await startApi();
    else await stopApi();
    return mcpInfo();
  });

  ipcMain.handle('chooseFfmpegPath', async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Choose the ffmpeg program (ffprobe should sit next to it)',
      properties: ['openFile'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    setFfmpegOverride(result.filePaths[0]);
    ffmpegCache = null;
    return mcpInfo();
  });

  ipcMain.handle('resetFfmpegPath', () => {
    setFfmpegOverride(null);
    ffmpegCache = null;
    return mcpInfo();
  });

  ipcMain.handle('getAppVersion', () => app.getVersion());
  ipcMain.handle('checkForUpdates', () => checkForUpdatesNow());
  ipcMain.handle('openHardpoint', () => openHardpoint());
  ipcMain.handle('hardpointIsReachable', () => isHardpointReachable());
}

function setupAutoUpdater(): void {
  if (!app.isPackaged) return;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on('update-downloaded', (info) => {
    void dialog
      .showMessageBox(mainWindow!, {
        type: 'info',
        title: 'Update ready',
        message: `KVGenius ${info.version} has been downloaded.`,
        detail: 'Restart now to install it, or it will install automatically the next time you quit.',
        buttons: ['Restart Now', 'Later'],
        defaultId: 0,
        cancelId: 1,
      })
      .then((result) => {
        if (result.response === 0) {
          autoUpdater.quitAndInstall();
        }
      });
  });

  autoUpdater.on('error', (err) => {
    console.error('Auto-update error:', err);
  });

  autoUpdater.checkForUpdates().catch((err) => {
    console.error('Failed to check for updates:', err);
  });
}

interface UpdateCheckResult {
  status: 'available' | 'not-available' | 'error' | 'unsupported';
  version?: string;
  message?: string;
}

function checkForUpdatesNow(): Promise<UpdateCheckResult> {
  if (!app.isPackaged) {
    return Promise.resolve({ status: 'unsupported' });
  }

  return new Promise((resolve) => {
    const cleanup = () => {
      autoUpdater.removeListener('update-available', onAvailable);
      autoUpdater.removeListener('update-not-available', onNotAvailable);
      autoUpdater.removeListener('error', onError);
    };
    const onAvailable = (info: { version: string }) => {
      cleanup();
      resolve({ status: 'available', version: info.version });
    };
    const onNotAvailable = () => {
      cleanup();
      resolve({ status: 'not-available' });
    };
    const onError = (err: Error) => {
      cleanup();
      const message = err?.message ?? String(err);
      // A CI release job uploads the installer before it generates/uploads the update
      // manifest (it needs the installer's own SHA512 first) -- a check that lands in that
      // multi-minute gap 404s on the manifest even though the release itself is live.
      resolve({
        status: 'error',
        message: message.includes('Cannot find latest')
          ? 'A new version may still be uploading -- try again in a few minutes.'
          : message,
      });
    };

    autoUpdater.once('update-available', onAvailable);
    autoUpdater.once('update-not-available', onNotAvailable);
    autoUpdater.once('error', onError);
    autoUpdater.checkForUpdates().catch(onError);
  });
}

app
  .whenReady()
  .then(async () => {
    migrateLegacyDefaultDbLocation();
    enforceDevDatabaseIsolation();
    db = initDatabase(getEffectiveDbPath());
    jobQueue = new JobQueue(db, createGenerationRunner(() => db), {
      cancelRunning: cancelCurrentGeneration,
      isCancellation: (err) => err instanceof GenerationCancelledError,
    });
    jobQueue.onProgress((_jobId, progress) => {
      mainWindow?.webContents.send('generationProgress', progress);
    });
    if (getApiEnabled()) {
      // A failure to start the API (say, a locked-down loopback) must not stop the app itself.
      await startApi().catch((error) => logStartupFailure('startApi', error));
    }
    moveLegacyOutput(db, videoFamilyList(), getLegacyOutputDir(), getImagesDir(), getVideosDir());
    // Favorited before the favorites folder existed: move those files into it.
    syncFavoriteFiles(db, outputDirs(), videoFamilyList());
    // The automatic cleanup does nothing unless the user turned it on in Settings (it is off by default).
    startCleanupSchedule({
      getDb: () => db,
      getSettings: getCleanupSettings,
      saveSettings: saveCleanupSettings,
      trashDir: getTrashDir,
      sourcesDir: getSourcesDir,
      recycle: recycleFile,
    });

    registerImageProtocol();
    mediaServer = await startMediaServer(mediaDirs, pickedSourceImages);
    // The preload script asks for this synchronously while the window is loading.
    ipcMain.on('getMediaBase', (event) => {
      event.returnValue = mediaServer?.base ?? '';
    });
    registerIpcHandlers();
    setupApplicationMenu();
    createWindow();
    setupAutoUpdater();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  })
  .catch((error) => {
    logStartupFailure('app.whenReady', error);
    showStartupFailureDialog(error);
    app.quit();
  });

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  removeDiscoveryFile(discoveryFilePath());
  void localApi?.close();
  localApi = null;
  mediaServer?.close();
  jobQueue = null;
  db?.close();
  db = null;
});
