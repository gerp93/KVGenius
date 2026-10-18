/** A run of matching text: `start` is the first character, `end` is one past the last. */
export interface TextMatch {
  start: number;
  end: number;
}

/** Every place `query` appears in `text`, left to right, ignoring upper/lower case. Matches never
 * overlap (searching "aa" in "aaa" finds one). An empty query finds nothing. */
export function findMatches(text: string, query: string): TextMatch[] {
  if (query === '') return [];
  const haystack = text.toLowerCase();
  const needle = query.toLowerCase();
  // Lower-casing can change a string's length for a few characters; then the positions would drift.
  if (haystack.length !== text.length || needle.length !== query.length) return findMatchesExact(text, query);
  const matches: TextMatch[] = [];
  let from = 0;
  for (;;) {
    const at = haystack.indexOf(needle, from);
    if (at === -1) return matches;
    matches.push({ start: at, end: at + needle.length });
    from = at + needle.length;
  }
}

/** The fallback for text whose lower-case form is a different length: an exact-case search. */
function findMatchesExact(text: string, query: string): TextMatch[] {
  const matches: TextMatch[] = [];
  let from = 0;
  for (;;) {
    const at = text.indexOf(query, from);
    if (at === -1) return matches;
    matches.push({ start: at, end: at + query.length });
    from = at + query.length;
  }
}

/** The match to step to from `current` (-1 when none is chosen yet), wrapping at either end.
 * Returns -1 when there are no matches. */
export function stepMatch(current: number, count: number, direction: 1 | -1): number {
  if (count <= 0) return -1;
  if (current < 0 || current >= count) return direction === 1 ? 0 : count - 1;
  return (current + direction + count) % count;
}

/** The first match at or after `caret`, so a search starts from where the cursor is; the first match
 * overall when the caret is past the last one. -1 when there are no matches. */
export function matchNearCaret(matches: TextMatch[], caret: number): number {
  if (matches.length === 0) return -1;
  const at = matches.findIndex((m) => m.end > caret);
  return at === -1 ? 0 : at;
}
