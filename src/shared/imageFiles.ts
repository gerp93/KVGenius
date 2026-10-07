/** The picture formats the app accepts as a source (video source image, Tools > Upscale). */
export const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp'];

/** Whether a file name (or path) ends in one of the accepted picture extensions, any letter case. */
export function isImageFileName(name: string): boolean {
  const dot = name.lastIndexOf('.');
  if (dot < 0) return false;
  return IMAGE_EXTENSIONS.includes(name.slice(dot + 1).toLowerCase());
}
