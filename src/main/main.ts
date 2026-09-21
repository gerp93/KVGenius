import { app, BrowserWindow, ipcMain, dialog, protocol, shell } from 'electron';
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
  getLegacyOutputDir,
  dbPathInsideFolder,
  enforceDevDatabaseIsolation,
  migrateLegacyDefaultDbLocation,
  getEffectiveComfyUIHost,
  setComfyUIHost,
  resetComfyUIHost,
  getEffectiveTheme,
  setTheme,
} from './dbLocation';
import {
  initDatabase,
  insertGeneration,
  listGenerations,
  countGenerations,
  listGenerationRefs,
  moveLegacyOutput,
  deleteGeneration,
  listSavedPrompts,
  insertSavedPrompt,
  updateSavedPrompt,
  deleteSavedPrompt,
} from './db';
import {
  generate as comfyGenerate,
  isAvailable as comfyIsAvailable,
  cancelCurrentGeneration,
  DEFAULT_COMFYUI_HOST,
} from './comfyui';
import { FAMILY_KIND, GenerationKind, GenerationParams } from '../shared/types';
import { isHardpointReachable, openHardpoint } from './hardpointLaunch';
import { MEDIA_SCHEME, MEDIA_SCHEME_PRIVILEGES, VIDEO_EXTENSIONS, handleMediaRequest } from './mediaProtocol';
import { MediaServer, startMediaServer } from './mediaServer';
import { faststartMp4 } from './mp4Faststart';
import { applyFavorite, syncFavoriteFiles } from './favorites';
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
  return { images: getImagesDir(), videos: getVideosDir(), legacy: getLegacyOutputDir() };
}

function videoFamilyList(): string[] {
  return Object.keys(FAMILY_KIND).filter((family) => FAMILY_KIND[family] === 'video');
}

function registerImageProtocol(): void {
  protocol.handle(MEDIA_SCHEME, (request) => handleMediaRequest(request, [getImagesDir(), getVideosDir(), getLegacyOutputDir()], pickedSourceImages));
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

function registerIpcHandlers(): void {
  ipcMain.handle('generate', async (_event, family: string, params: GenerationParams) => {
    if (!db) throw new Error('Database not initialized');

    const output = await comfyGenerate(family, params);

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

    const record = insertGeneration(db, params, family, imagePath);
    return { record, imageUrl: imageUrlFor(imagePath) };
  });

  ipcMain.handle('cancelGeneration', () => cancelCurrentGeneration());

  const videoFamilies = videoFamilyList();

  ipcMain.handle(
    'listGenerations',
    (_event, kind: GenerationKind, limit: number, beforeId: number | null, favoritesOnly: boolean) => {
      if (!db) throw new Error('Database not initialized');
      const safeLimit = Math.min(Math.max(Math.floor(limit) || 0, 1), 200);
      return listGenerations(db, videoFamilies, kind === 'video' ? 'video' : 'image', safeLimit, beforeId, !!favoritesOnly);
    }
  );

  ipcMain.handle('countGenerations', (_event, favoritesOnly: boolean) => {
    if (!db) throw new Error('Database not initialized');
    return countGenerations(db, videoFamilies, !!favoritesOnly);
  });

  ipcMain.handle('listGenerationRefs', (_event, kind: GenerationKind, favoritesOnly: boolean) => {
    if (!db) throw new Error('Database not initialized');
    return listGenerationRefs(db, videoFamilies, kind === 'video' ? 'video' : 'image', !!favoritesOnly);
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

  ipcMain.handle('deleteGeneration', (_event, id: number, imagePath: string) => {
    if (!db) throw new Error('Database not initialized');
    deleteGeneration(db, id);
    try {
      fs.unlinkSync(imagePath);
    } catch {
      // Already gone (or never existed) - the DB row is still correctly deleted either way.
    }
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

  ipcMain.handle('listSavedPrompts', () => {
    if (!db) throw new Error('Database not initialized');
    return listSavedPrompts(db);
  });

  ipcMain.handle('savePrompt', (_event, name: string, prompt: string, tags: string[]) => {
    if (!db) throw new Error('Database not initialized');
    return insertSavedPrompt(db, name, prompt, tags);
  });

  ipcMain.handle('updateSavedPrompt', (_event, id: number, name: string, tags: string[]) => {
    if (!db) throw new Error('Database not initialized');
    return updateSavedPrompt(db, id, name, tags);
  });

  ipcMain.handle('deleteSavedPrompt', (_event, id: number) => {
    if (!db) throw new Error('Database not initialized');
    deleteSavedPrompt(db, id);
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

  ipcMain.handle('getDbInfo', () => ({
    path: getEffectiveDbPath(),
    isDefault: isUsingDefaultDbLocation(),
    defaultPath: getDefaultDbPath(),
  }));

  ipcMain.handle('revealDbInFileManager', () => revealDbInFileManager());

  // The database must be closed before its file is copied/adopted (setDbPath's job), and a
  // live node:sqlite connection can't just be repointed at a different path afterward - the
  // simplest correct fix is a full relaunch, which re-opens at whatever getEffectiveDbPath()
  // now resolves to. Matches the standard's own "then restart the app" requirement.
  function relocateAndRelaunch(newPath: string): void {
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
    db?.close();
    db = null;
    resetToDefaultDbPath();
    app.relaunch();
    app.exit();
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
    moveLegacyOutput(db, videoFamilyList(), getLegacyOutputDir(), getImagesDir(), getVideosDir());
    // Favorited before the favorites folder existed: move those files into it.
    syncFavoriteFiles(db, outputDirs(), videoFamilyList());

    registerImageProtocol();
    mediaServer = await startMediaServer(
      () => [getImagesDir(), getVideosDir(), getLegacyOutputDir()],
      pickedSourceImages
    );
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
  mediaServer?.close();
  db?.close();
  db = null;
});
