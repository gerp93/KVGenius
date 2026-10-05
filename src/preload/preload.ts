import { contextBridge, ipcRenderer } from 'electron';
import { GenerationKind, GenerationParams, GenerationProgress, KVGeniusAPI } from '../shared/types';
import { PromptSlot } from '../shared/promptSlots';

// Videos are served by a local HTTP server, everything else by the kvimage:// protocol - the same
// rule as imageUrlFor() in main.ts (the sandboxed preload can't import it).
const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.mkv'];
const mediaBase: string = ipcRenderer.sendSync('getMediaBase');

function mediaUrlFor(filePath: string): string {
  const isVideo = VIDEO_EXTENSIONS.some((ext) => filePath.toLowerCase().endsWith(ext));
  return isVideo && mediaBase ? `${mediaBase}/${encodeURIComponent(filePath)}` : `kvimage://${encodeURIComponent(filePath)}`;
}

const api: KVGeniusAPI = {
  generate: (family: string, params: GenerationParams, estimate?: { totalMs: number | null; generateMs: number | null } | null) =>
    ipcRenderer.invoke('generate', family, params, estimate ?? null),
  estimateGeneration: (family: string, params: GenerationParams, previousFamily?: string | null) =>
    ipcRenderer.invoke('estimateGeneration', family, params, previousFamily),
  onGenerationProgress: (callback: (progress: GenerationProgress) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, progress: GenerationProgress) => callback(progress);
    ipcRenderer.on('generationProgress', listener);
    return () => ipcRenderer.removeListener('generationProgress', listener);
  },
  getTimingStats: () => ipcRenderer.invoke('getTimingStats'),
  clearTimingStats: () => ipcRenderer.invoke('clearTimingStats'),
  cancelGeneration: () => ipcRenderer.invoke('cancelGeneration'),
  listGenerations: (
    kind: GenerationKind,
    limit: number,
    beforeId: number | null,
    favoritesOnly: boolean,
    showHidden: boolean,
    extension?: string | null
  ) => ipcRenderer.invoke('listGenerations', kind, limit, beforeId, favoritesOnly, showHidden, extension ?? null),
  countGenerations: (favoritesOnly: boolean, showHidden: boolean, imageExtension?: string | null) =>
    ipcRenderer.invoke('countGenerations', favoritesOnly, showHidden, imageExtension ?? null),
  listImageExtensions: () => ipcRenderer.invoke('listImageExtensions'),
  listGenerationRefs: (kind: GenerationKind, favoritesOnly: boolean, showHidden: boolean, extension?: string | null) =>
    ipcRenderer.invoke('listGenerationRefs', kind, favoritesOnly, showHidden, extension ?? null),
  setGenerationHidden: (id: number, hidden: boolean) => ipcRenderer.invoke('setGenerationHidden', id, hidden),
  getHiddenWords: () => ipcRenderer.invoke('getHiddenWords'),
  setHiddenWords: (words: string[]) => ipcRenderer.invoke('setHiddenWords', words),
  applyHiddenWords: () => ipcRenderer.invoke('applyHiddenWords'),
  getFileSize: (imagePath: string) => ipcRenderer.invoke('getFileSize', imagePath),
  exportGenerations: (imagePaths: string[]) => ipcRenderer.invoke('exportGenerations', imagePaths),
  setGenerationFavorite: (id: number, favorite: boolean) => ipcRenderer.invoke('setGenerationFavorite', id, favorite),
  imageUrlFor: (imagePath: string) => mediaUrlFor(imagePath),
  chooseSourceImage: () => ipcRenderer.invoke('chooseSourceImage'),
  copyImageToClipboard: (imagePath: string) => ipcRenderer.invoke('copyImageToClipboard', imagePath),
  revealGenerationInFileManager: (imagePath: string) => ipcRenderer.invoke('revealGenerationInFileManager', imagePath),
  diagnoseVideo: (imagePath: string) => ipcRenderer.invoke('diagnoseVideo', imagePath),
  openGenerationExternally: (imagePath: string) => ipcRenderer.invoke('openGenerationExternally', imagePath),
  saveGenerationAs: (imagePath: string) => ipcRenderer.invoke('saveGenerationAs', imagePath),

  setGenerationPinned: (id: number, pinned: boolean) => ipcRenderer.invoke('setGenerationPinned', id, pinned),
  listPinnedGenerations: (showHidden: boolean) => ipcRenderer.invoke('listPinnedGenerations', showHidden),

  getCleanupSettings: () => ipcRenderer.invoke('getCleanupSettings'),
  setCleanupSettings: (patch: {
    autoTrashEnabled?: boolean;
    olderThanDays?: number;
    autoEmptyEnabled?: boolean;
    trashRetentionDays?: number;
  }) => ipcRenderer.invoke('setCleanupSettings', patch),
  previewCleanup: (days: number) => ipcRenderer.invoke('previewCleanup', days),
  runCleanup: (days: number) => ipcRenderer.invoke('runCleanup', days),
  trashGenerations: (ids: number[], options?: { includeKept?: boolean }) => ipcRenderer.invoke('trashGenerations', ids, options),
  getTrashStats: () => ipcRenderer.invoke('getTrashStats'),
  listTrashed: (limit: number, beforeId: number | null) => ipcRenderer.invoke('listTrashed', limit, beforeId),
  restoreGenerations: (ids: number[]) => ipcRenderer.invoke('restoreGenerations', ids),
  deleteTrashed: (ids: number[]) => ipcRenderer.invoke('deleteTrashed', ids),
  emptyTrash: () => ipcRenderer.invoke('emptyTrash'),

  getPromptSlots: () => ipcRenderer.invoke('getPromptSlots'),
  savePromptSlots: (slots: PromptSlot[], activeId: string) => ipcRenderer.invoke('savePromptSlots', slots, activeId),

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

  launchComfyUI: () => ipcRenderer.invoke('launchComfyUI'),
  getComfyUILauncher: () => ipcRenderer.invoke('getComfyUILauncher'),
  chooseComfyUILauncher: () => ipcRenderer.invoke('chooseComfyUILauncher'),
  clearComfyUILauncher: () => ipcRenderer.invoke('clearComfyUILauncher'),
  getMcpInfo: () => ipcRenderer.invoke('getMcpInfo'),
  setMcpEnabled: (enabled: boolean) => ipcRenderer.invoke('setMcpEnabled', enabled),
  chooseFfmpegPath: () => ipcRenderer.invoke('chooseFfmpegPath'),
  resetFfmpegPath: () => ipcRenderer.invoke('resetFfmpegPath'),
  getAppVersion: () => ipcRenderer.invoke('getAppVersion'),
  checkForUpdates: () => ipcRenderer.invoke('checkForUpdates'),

  openHardpoint: () => ipcRenderer.invoke('openHardpoint'),
  hardpointIsReachable: () => ipcRenderer.invoke('hardpointIsReachable'),
  listUpscaleModels: () => ipcRenderer.invoke('listUpscaleModels'),
  convertToGif: (id: number, options: { fps: number; width: number }) => ipcRenderer.invoke('convertToGif', id, options),
};

contextBridge.exposeInMainWorld('kvgenius', api);
