/** Model family recorded for a GIF made from a video. Deliberately not in FAMILY_KIND, so the
 * Library files it under Images (an <img> plays a GIF) and outside clients cannot generate with it. */
export const GIF_FAMILY = 'video-gif';

/** Widest the GIF is made (never enlarged past the video's own width). GIFs grow fast with size. */
export const GIF_WIDTHS = [320, 480, 640, 800];
export const DEFAULT_GIF_WIDTH = 480;

export const GIF_FPS_CHOICES = [8, 10, 12, 15, 20];
export const DEFAULT_GIF_FPS = 15;
