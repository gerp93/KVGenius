import { contextBridge, ipcRenderer } from 'electron';
import { GenerationParams, KVGeniusAPI } from '../shared/types';

const api: KVGeniusAPI = {
  generate: (params: GenerationParams) => ipcRenderer.invoke('generate', params),
  listGenerations: () => ipcRenderer.invoke('listGenerations'),
  imageUrlFor: (imagePath: string) => `kvimage://${encodeURIComponent(imagePath)}`,

  listSavedPrompts: () => ipcRenderer.invoke('listSavedPrompts'),
  savePrompt: (name: string | null, prompt: string) => ipcRenderer.invoke('savePrompt', name, prompt),
  deleteSavedPrompt: (id: number) => ipcRenderer.invoke('deleteSavedPrompt', id),
};

contextBridge.exposeInMainWorld('kvgenius', api);
