export const MAX_HIDDEN_WORDS = 500;
export const MAX_HIDDEN_WORD_LENGTH = 60;

/** Cleans the word list: lowercased, spaces collapsed, no empties or duplicates, each at most
 * MAX_HIDDEN_WORD_LENGTH long, at most MAX_HIDDEN_WORDS of them. A word may be a phrase. */
export function normalizeHiddenWords(words: readonly string[] | null | undefined): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const raw of words ?? []) {
    const word = String(raw).trim().toLowerCase().replace(/\s+/g, ' ').slice(0, MAX_HIDDEN_WORD_LENGTH).trim();
    if (!word || seen.has(word)) continue;
    seen.add(word);
    result.push(word);
    if (result.length >= MAX_HIDDEN_WORDS) break;
  }
  return result;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Builds a test for "does this prompt contain any of these words". Matches whole words only (so
 * "ass" does not hide "class"), ignoring case, allowing a plural "s"/"es" on the end, and treating
 * any run of spaces inside a phrase as one. Returns a function that is always false for an empty list.
 */
export function compileHiddenMatcher(words: readonly string[]): (prompt: string) => boolean {
  const cleaned = normalizeHiddenWords(words);
  if (cleaned.length === 0) return () => false;
  const alternatives = cleaned.map((w) => escapeRegExp(w).replace(/ /g, '\\s+')).join('|');
  const pattern = new RegExp(`(?<![\\p{L}\\p{N}])(?:${alternatives})(?:e?s)?(?![\\p{L}\\p{N}])`, 'iu');
  return (prompt) => pattern.test(prompt);
}
