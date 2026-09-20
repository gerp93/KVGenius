import { app, BrowserWindow, ipcMain, dialog, protocol, net, shell } from 'electron';
import { DatabaseSync } from 'node:sqlite';
import * as path from 'path';
import * as fs from 'fs';
import { pathToFileURL } from 'url';
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
  deleteGeneration,
  listSavedPrompts,
  insertSavedPrompt,
  deleteSavedPrompt,
} from './db';
import {
  generate as comfyGenerate,
  isAvailable as comfyIsAvailable,
  cancelCurrentGeneration,
  DEFAULT_COMFYUI_HOST,
} from './comfyui';
import { GenerationParams } from '../shared/types';

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
  { scheme: 'kvimage', privileges: { secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } },
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

function registerImageProtocol(): void {
  protocol.handle('kvimage', (request) => {
    const encodedPath = request.url.replace('kvimage://', '');
    const filePath = decodeURIComponent(encodedPath);
    // Only ever serve files inside the app's own images directory - the renderer passes
    // paths back that originated from the database, but this is cheap insurance against a
    // malformed/crafted kvimage:// URL reaching outside it.
    const imagesDir = path.resolve(getImagesDir());
    const resolved = path.resolve(filePath);
    if (!resolved.startsWith(imagesDir)) {
      return new Response('Forbidden', { status: 403 });
    }
    return net.fetch(pathToFileURL(resolved).toString());
  });
}

function imageUrlFor(imagePath: string): string {
  return `kvimage://${encodeURIComponent(imagePath)}`;
}

function registerIpcHandlers(): void {
  ipcMain.handle('generate', async (_event, family: string, params: GenerationParams) => {
    if (!db) throw new Error('Database not initialized');

    const output = await comfyGenerate(family, params);

    const imagesDir = getImagesDir();
    fs.mkdirSync(imagesDir, { recursive: true });
    const filename = `${Date.now()}-${params.seed}${output.extension}`;
    const imagePath = path.join(imagesDir, filename);
    fs.writeFileSync(imagePath, output.bytes);

    const record = insertGeneration(db, params, family, imagePath);
    return { record, imageUrl: imageUrlFor(imagePath) };
  });

  ipcMain.handle('cancelGeneration', () => cancelCurrentGeneration());

  ipcMain.handle('listGenerations', () => {
    if (!db) throw new Error('Database not initialized');
    return listGenerations(db);
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

  ipcMain.handle('savePrompt', (_event, name: string | null, prompt: string) => {
    if (!db) throw new Error('Database not initialized');
    return insertSavedPrompt(db, name, prompt);
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
  .then(() => {
    migrateLegacyDefaultDbLocation();
    enforceDevDatabaseIsolation();
    db = initDatabase(getEffectiveDbPath());

    registerImageProtocol();
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
  db?.close();
  db = null;
});
