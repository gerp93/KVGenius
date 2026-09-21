import * as path from 'path';
import * as fs from 'fs';
import { app, shell, dialog } from 'electron';
import { DEFAULT_COMFYUI_HOST } from './comfyui';

interface AppConfig {
  dbPath?: string;
  comfyuiHost?: string;
  theme?: string;
}

export const DEFAULT_THEME = 'neon';

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

// The db always lives inside its own app-named subfolder, both at the default install
// location and anywhere it's relocated to - not a bare file/`images` sibling directly in a
// shared parent directory, which silently collides with another app's same-named db/images
// folder if the user ever points both apps at a shared parent (e.g. a synced backup folder).
// Suffixed 'Data' (not just 'KVGenius') because at the default install location this folder
// nests inside userData, which Electron already names after the app (app.setName('kvgenius'))
// - 'kvgenius/KVGenius/' would be a redundant identically-named parent/child pair.
const DB_SUBFOLDER = 'KVGenius_Data';
const DB_FILENAME = 'kvgenius.db';

/** Where the installed app would open its database - used to keep dev from sharing it. */
export function getPackagedConfiguredDbPath(): string | null {
  if (app.isPackaged) return null;
  const packagedConfig = readConfigAt(getPackagedAppConfigPath());
  if (packagedConfig.dbPath?.trim()) return normalizeDbPath(packagedConfig.dbPath);
  return normalizeDbPath(path.join(app.getPath('appData'), 'kvgenius', DB_SUBFOLDER, DB_FILENAME));
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
  return path.join(app.getPath('userData'), DB_SUBFOLDER, DB_FILENAME);
}

/** Where the db file would live if relocated inside the given parent folder - always nested
 * under DB_SUBFOLDER, same as the default location, so a user picking a shared parent folder
 * (e.g. one also used by another app) can't collide with that other app's own db/images. */
export function dbPathInsideFolder(parentFolder: string): string {
  return path.join(parentFolder, DB_SUBFOLDER, DB_FILENAME);
}

/**
 * One-time migration for installs that started before the default location was nested under
 * DB_SUBFOLDER: if nothing has ever been manually relocated, the new nested default doesn't
 * exist yet, but the old flat `userData/kvgenius.db` does, move it (and its WAL/SHM sidecars,
 * if present) into place. Must run before initDatabase() ever opens a connection.
 *
 * Existing generated images are deliberately NOT moved here - their DB rows store absolute
 * paths and keep working exactly where they are; only new generations start landing in the
 * newly-nested images folder (getImagesDir() is always a sibling of wherever the db lives).
 */
export function migrateLegacyDefaultDbLocation(): void {
  if (!isUsingDefaultDbLocation()) return;

  const newPath = getDefaultDbPath();
  if (fs.existsSync(newPath)) return;

  const oldPath = path.join(app.getPath('userData'), DB_FILENAME);
  if (!fs.existsSync(oldPath)) return;

  fs.mkdirSync(path.dirname(newPath), { recursive: true });
  fs.renameSync(oldPath, newPath);
  for (const suffix of ['-wal', '-shm']) {
    const oldSidecar = oldPath + suffix;
    if (fs.existsSync(oldSidecar)) fs.renameSync(oldSidecar, newPath + suffix);
  }
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

/** Parent of all generated output, next to the database: KVGenius_Data/output. */
function getOutputDir(): string {
  return path.join(path.dirname(getEffectiveDbPath()), 'output');
}

/** Where generated images are saved: KVGenius_Data/output/images. */
export function getImagesDir(): string {
  return path.join(getOutputDir(), 'images');
}

/** Where generated videos are saved: KVGenius_Data/output/videos. */
export function getVideosDir(): string {
  return path.join(getOutputDir(), 'videos');
}

/** Before output was grouped under `output/`, both kinds were saved in KVGenius_Data/images. */
export function getLegacyOutputDir(): string {
  return path.join(path.dirname(getEffectiveDbPath()), 'images');
}

export function getEffectiveTheme(): string {
  return readConfig().theme || DEFAULT_THEME;
}

export function setTheme(themeId: string): void {
  writeConfig({ ...readConfig(), theme: themeId.trim() });
}
