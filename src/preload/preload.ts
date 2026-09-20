import { contextBridge, ipcRenderer } from 'electron';
import { GenerationParams, KVGeniusAPI } from '../shared/types';

const api: KVGeniusAPI = {
  generate: (params: GenerationParams) => ipcRenderer.invoke('generate', params),
  listGenerations: () => ipcRenderer.invoke('listGenerations'),
  imageUrlFor: (imagePath: string) => `kvimage://${encodeURIComponent(imagePath)}`,

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
};

contextBridge.exposeInMainWorld('kvgenius', api);
