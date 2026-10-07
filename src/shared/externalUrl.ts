/** Only web links may be handed to the OS browser: the renderer is asked to open them, and a
 * `file:` or custom-protocol URL would run something other than a web page. */
export function isExternalWebUrl(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:';
  } catch {
    return false;
  }
}
