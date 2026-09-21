import { contextBridge, ipcRenderer } from 'electron';
import { GenerationKind, GenerationParams, KVGeniusAPI } from '../shared/types';

// Videos are served by a local HTTP server, everything else by the kvimage:// protocol - the same
// rule as imageUrlFor() in main.ts (the sandboxed preload can't import it).
const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.mkv'];
const mediaBase: string = ipcRenderer.sendSync('getMediaBase');

function mediaUrlFor(filePath: string): string {
  const isVideo = VIDEO_EXTENSIONS.some((ext) => filePath.toLowerCase().endsWith(ext));
  return isVideo && mediaBase ? `${mediaBase}/${encodeURIComponent(filePath)}` : `kvimage://${encodeURIComponent(filePath)}`;
}

const api: KVGeniusAPI = {
  generate: (family: string, params: GenerationParams) => ipcRenderer.invoke('generate', family, params),
  cancelGeneration: () => ipcRenderer.invoke('cancelGeneration'),
  listGenerations: (kind: GenerationKind, limit: number, beforeId: number | null, favoritesOnly: boolean) =>
    ipcRenderer.invoke('listGenerations', kind, limit, beforeId, favoritesOnly),
  countGenerations: (favoritesOnly: boolean) => ipcRenderer.invoke('countGenerations', favoritesOnly),
  listGenerationRefs: (kind: GenerationKind, favoritesOnly: boolean) => ipcRenderer.invoke('listGenerationRefs', kind, favoritesOnly),
  getFileSize: (imagePath: string) => ipcRenderer.invoke('getFileSize', imagePath),
  exportGenerations: (imagePaths: string[]) => ipcRenderer.invoke('exportGenerations', imagePaths),
  setGenerationFavorite: (id: number, favorite: boolean) => ipcRenderer.invoke('setGenerationFavorite', id, favorite),
  imageUrlFor: (imagePath: string) => mediaUrlFor(imagePath),
  chooseSourceImage: () => ipcRenderer.invoke('chooseSourceImage'),
  deleteGeneration: (id: number, imagePath: string) => ipcRenderer.invoke('deleteGeneration', id, imagePath),
  revealGenerationInFileManager: (imagePath: string) => ipcRenderer.invoke('revealGenerationInFileManager', imagePath),
  diagnoseVideo: (imagePath: string) => ipcRenderer.invoke('diagnoseVideo', imagePath),
  openGenerationExternally: (imagePath: string) => ipcRenderer.invoke('openGenerationExternally', imagePath),
  saveGenerationAs: (imagePath: string) => ipcRenderer.invoke('saveGenerationAs', imagePath),

  listSavedPrompts: () => ipcRenderer.invoke('listSavedPrompts'),
  savePrompt: (name: string, prompt: string, tags: string[]) => ipcRenderer.invoke('savePrompt', name, prompt, tags),
  updateSavedPrompt: (id: number, name: string, tags: string[]) => ipcRenderer.invoke('updateSavedPrompt', id, name, tags),
  deleteSavedPrompt: (id: number) => ipcRenderer.invoke('deleteSavedPrompt', id),

  getComfyUIHost: () => ipcRenderer.invoke('getComfyUIHost'),
  setComfyUIHost: (host: string) => ipcRenderer.invoke('setComfyUIHost', host),
  resetComfyUIHost: () => ipcRenderer.invoke('resetComfyUIHost'),
  checkComfyUIConnection: () => ipcRenderer.invoke('checkComfyUIConnection'),

  getTheme: () => ipcRenderer.invoke('getTheme'),
  setTheme: (themeId: string) => ipcRenderer.invoke('setTheme', themeId),

  getDbInfo: () => ipcRenderer.invoke('getDbInfo'),
  revealDbInFileManager: () => ipcRenderer.invoke('revealDbInFileManager'),
  chooseExistingDb: () => ipcRenderer.invoke('chooseExistingDb'),
  chooseNewDbLocation: () => ipcRenderer.invoke('chooseNewDbLocation'),
  resetDbToDefault: () => ipcRenderer.invoke('resetDbToDefault'),

  getAppVersion: () => ipcRenderer.invoke('getAppVersion'),
  checkForUpdates: () => ipcRenderer.invoke('checkForUpdates'),

  openHardpoint: () => ipcRenderer.invoke('openHardpoint'),
  hardpointIsReachable: () => ipcRenderer.invoke('hardpointIsReachable'),
};

contextBridge.exposeInMainWorld('kvgenius', api);
