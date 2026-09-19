import { app, BrowserWindow, ipcMain, dialog, protocol, net } from 'electron';
import { DatabaseSync } from 'node:sqlite';
import * as path from 'path';
import * as fs from 'fs';
import { pathToFileURL } from 'url';
import { autoUpdater } from 'electron-updater';

import { setupApplicationMenu, attachContextMenu } from './menu';
import {
  getEffectiveDbPath,
  getImagesDir,
  enforceDevDatabaseIsolation,
} from './dbLocation';
import { initDatabase, insertGeneration, listGenerations, listSavedPrompts, insertSavedPrompt, deleteSavedPrompt } from './db';
import { generate as comfyGenerate } from './comfyui';
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
    void mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
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
  ipcMain.handle('generate', async (_event, params: GenerationParams) => {
    if (!db) throw new Error('Database not initialized');

    const imageBytes = await comfyGenerate('z-image-turbo', params);

    const imagesDir = getImagesDir();
    fs.mkdirSync(imagesDir, { recursive: true });
    const filename = `${Date.now()}-${params.seed}.png`;
    const imagePath = path.join(imagesDir, filename);
    fs.writeFileSync(imagePath, imageBytes);

    const record = insertGeneration(db, params, 'z-image-turbo', imagePath);
    return { record, imageUrl: imageUrlFor(imagePath) };
  });

  ipcMain.handle('listGenerations', () => {
    if (!db) throw new Error('Database not initialized');
    return listGenerations(db);
  });

  ipcMain.handle('imageUrlFor', (_event, imagePath: string) => imageUrlFor(imagePath));

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
}

app
  .whenReady()
  .then(() => {
    enforceDevDatabaseIsolation();
    db = initDatabase(getEffectiveDbPath());

    registerImageProtocol();
    registerIpcHandlers();
    setupApplicationMenu();
    createWindow();

    if (app.isPackaged) {
      void autoUpdater.checkForUpdatesAndNotify();
    }

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
