import * as path from 'path';
import * as fs from 'fs';
import { app, shell, dialog } from 'electron';
import { DEFAULT_COMFYUI_HOST } from './comfyui';

interface AppConfig {
  dbPath?: string;
  comfyuiHost?: string;
}

function getConfigPath(): string {
  return path.join(app.getPath('userData'), 'app-config.json');
}

function readConfigAt(configPath: string): AppConfig {
  if (!fs.existsSync(configPath)) return {};
  try {
    return JSON.parse(fs.readFileSync(configPath, 'utf-8').replace(/^﻿/, ''));
  } catch {
    return {};
  }
}

function readConfig(): AppConfig {
  return readConfigAt(getConfigPath());
}

function writeConfig(config: AppConfig): void {
  fs.mkdirSync(path.dirname(getConfigPath()), { recursive: true });
  fs.writeFileSync(getConfigPath(), JSON.stringify(config, null, 2));
}

function normalizeDbPath(filePath: string): string {
  const resolved = path.resolve(filePath.trim());
  return process.platform === 'win32' ? path.win32.normalize(resolved) : path.normalize(resolved);
}

export function getPackagedAppConfigPath(): string {
  return path.join(app.getPath('appData'), 'kvgenius', 'app-config.json');
}

/** Where the installed app would open its database - used to keep dev from sharing it. */
export function getPackagedConfiguredDbPath(): string | null {
  if (app.isPackaged) return null;
  const packagedConfig = readConfigAt(getPackagedAppConfigPath());
  if (packagedConfig.dbPath?.trim()) return normalizeDbPath(packagedConfig.dbPath);
  return normalizeDbPath(path.join(app.getPath('appData'), 'kvgenius', 'kvgenius.db'));
}

export function isPackagedDatabasePath(dbPath: string): boolean {
  const packagedPath = getPackagedConfiguredDbPath();
  return packagedPath !== null && normalizeDbPath(dbPath) === packagedPath;
}

/** Dev must not open the packaged app's database. Reset to the dev default if it does. */
export function enforceDevDatabaseIsolation(): boolean {
  if (app.isPackaged || !isPackagedDatabasePath(getEffectiveDbPath())) return false;

  resetToDefaultDbPath();
  dialog.showMessageBoxSync({
    type: 'warning',
    title: 'Dev database reset',
    message: 'Dev was pointed at the packaged app database.',
    detail: `Dev now uses its isolated default:\n${getDefaultDbPath()}`,
    buttons: ['OK'],
  });
  return true;
}

export function getDefaultDbPath(): string {
  return path.join(app.getPath('userData'), 'kvgenius.db');
}

/** The database file the app will actually load on startup: a user-chosen location, or the default. */
export function getEffectiveDbPath(): string {
  const configured = readConfig().dbPath;
  return configured && configured.trim() !== '' ? configured : getDefaultDbPath();
}

export function isUsingDefaultDbLocation(): boolean {
  return !readConfig().dbPath;
}

/**
 * Point the app at a different SQLite file. If nothing exists yet at the new
 * location, the current database is copied there first so no data is lost.
 * If a file already exists there, it's left alone and adopted as-is.
 *
 * PRECONDITION: the database must already be closed (node:sqlite's DatabaseSync
 * needs no WAL checkpoint step the way better-sqlite3/sql.js setups do here,
 * but the file must not be open for write while it's being copied).
 */
export function setDbPath(newPath: string): void {
  if (!app.isPackaged && isPackagedDatabasePath(newPath)) {
    throw new Error('Dev cannot use the packaged app database. Choose a different file.');
  }

  const currentPath = getEffectiveDbPath();
  if (!fs.existsSync(newPath) && fs.existsSync(currentPath)) {
    fs.mkdirSync(path.dirname(newPath), { recursive: true });
    fs.copyFileSync(currentPath, newPath);
  }

  writeConfig({ ...readConfig(), dbPath: newPath });
}

export function resetToDefaultDbPath(): void {
  const config = readConfig();
  delete config.dbPath;
  writeConfig(config);
}

/** Open the system file manager at the database file, or its parent folder if missing. */
export async function revealDbInFileManager(): Promise<void> {
  const dbPath = normalizeDbPath(getEffectiveDbPath());
  if (fs.existsSync(dbPath)) {
    shell.showItemInFolder(dbPath);
    return;
  }
  const dir = normalizeDbPath(path.dirname(dbPath));
  if (!fs.existsSync(dir)) throw new Error(`Database folder not found: ${dir}`);
  const err = await shell.openPath(dir);
  if (err) throw new Error(err);
}

export function getEffectiveComfyUIHost(): string {
  const configured = readConfig().comfyuiHost;
  return configured && configured.trim() !== '' ? configured.trim() : DEFAULT_COMFYUI_HOST;
}

export function isUsingDefaultComfyUIHost(): boolean {
  return !readConfig().comfyuiHost;
}

export function setComfyUIHost(host: string): void {
  writeConfig({ ...readConfig(), comfyuiHost: host.trim() });
}

export function resetComfyUIHost(): void {
  const config = readConfig();
  delete config.comfyuiHost;
  writeConfig(config);
}

/** Default folder generated images are saved into, next to the database. */
export function getImagesDir(): string {
  return path.join(path.dirname(getEffectiveDbPath()), 'images');
}
