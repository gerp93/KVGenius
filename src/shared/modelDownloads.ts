/** One file the download helper would fetch. */
export interface DownloadItemInfo {
  label: string;
  url: string;
  /** The ComfyUI folder it goes in (diffusion_models, vae, ...). */
  folder: string;
  fileName: string;
  /** Full path it will be saved to. */
  destPath: string;
}

/** What "Download missing files" would do right now, shown for confirmation before anything runs. */
export interface DownloadPlanInfo {
  items: DownloadItemInfo[];
  /** Files that exist but sit in a subfolder: not downloaded again, the user moves them. */
  inSubfolder: string[];
  /** Why nothing can be downloaded (no models folder, ...), or null. */
  problem: string | null;
}

export type DownloadStartResult =
  /** A terminal window was opened running the script. */
  | { status: 'launched'; scriptPath: string }
  /** No terminal could be opened: the script is given to run by hand. */
  | { status: 'copy'; scriptPath: string; script: string; reason: string }
  | { status: 'error'; message: string };
