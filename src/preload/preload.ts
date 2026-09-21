import { contextBridge, ipcRenderer } from 'electron';
import { GenerationKind, GenerationParams, KVGeniusAPI } from '../shared/types';

const api: KVGeniusAPI = {
  generate: (family: string, params: GenerationParams) => ipcRenderer.invoke('generate', family, params),
  cancelGeneration: () => ipcRenderer.invoke('cancelGeneration'),
  listGenerations: (kind: GenerationKind, limit: number, beforeId: number | null, favoritesOnly: boolean) =>
    ipcRenderer.invoke('listGenerations', kind, limit, beforeId, favoritesOnly),
  countGenerations: (favoritesOnly: boolean) => ipcRenderer.invoke('countGenerations', favoritesOnly),
  setGenerationFavorite: (id: number, favorite: boolean) => ipcRenderer.invoke('setGenerationFavorite', id, favorite),
  imageUrlFor: (imagePath: string) => `kvimage://${encodeURIComponent(imagePath)}`,
  chooseSourceImage: () => ipcRenderer.invoke('chooseSourceImage'),
  deleteGeneration: (id: number, imagePath: string) => ipcRenderer.invoke('deleteGeneration', id, imagePath),
  revealGenerationInFileManager: (imagePath: string) => ipcRenderer.invoke('revealGenerationInFileManager', imagePath),
  saveGenerationAs: (imagePath: string) => ipcRenderer.invoke('saveGenerationAs', imagePath),

  listSavedPrompts: () => ipcRenderer.invoke('listSavedPrompts'),
  savePrompt: (name: string | null, prompt: string) => ipcRenderer.invoke('savePrompt', name, prompt),
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
