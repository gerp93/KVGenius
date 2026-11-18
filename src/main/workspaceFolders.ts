import * as path from 'path';

/**
 * Where a workspace keeps its files: `output/<folder>/{images,videos,gifs,sources,trash}`. The folder name is chosen when the workspace is
 * made (from its name, see `folderNameFor`) and stored with it, so renaming a workspace later never moves a file.
 */
export interface WorkspaceDirs {
  /** The workspace's own folder. */
  root: string;
  images: string;
  videos: string;
  gifs: string;
  /** Kept copies of source images and masks, named by their contents. */
  sources: string;
  /** Items moved to the Trash keep their files here until restored or deleted for good. */
  trash: string;
}

/** The folders that sat directly under `output/` before workspaces had folders of their own. A workspace folder may not take these names. */
export const LEGACY_FOLDER_NAMES = ['images', 'videos', 'gifs', 'sources', 'trash'] as const;

export function workspaceDirs(outputRoot: string, folder: string): WorkspaceDirs {
  const root = path.join(outputRoot, folder);
  return {
    root,
    images: path.join(root, 'images'),
    videos: path.join(root, 'videos'),
    gifs: path.join(root, 'gifs'),
    sources: path.join(root, 'sources'),
    trash: path.join(root, 'trash'),
  };
}

const WINDOWS_RESERVED = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i;

/**
 * A folder name for a workspace: its name with anything a folder name cannot hold taken out, different (ignoring case) from every name in
 * `taken` and from the legacy folders. A name with nothing usable in it becomes "workspace".
 */
export function folderNameFor(workspaceName: string, taken: readonly string[]): string {
  let base = workspaceName
    // eslint-disable-next-line no-control-regex
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[. ]+|[. ]+$/g, '')
    .slice(0, 40)
    .trim();
  if (base === '') base = 'workspace';
  const used = new Set(taken.map((t) => t.toLowerCase()));
  const clashes = (name: string) =>
    used.has(name.toLowerCase()) || WINDOWS_RESERVED.test(name) || (LEGACY_FOLDER_NAMES as readonly string[]).includes(name.toLowerCase());
  if (!clashes(base)) return base;
  for (let n = 2; ; n++) {
    const candidate = `${base}-${n}`;
    if (!clashes(candidate)) return candidate;
  }
}
