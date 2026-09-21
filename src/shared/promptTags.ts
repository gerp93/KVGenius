export const MAX_NAME_LENGTH = 80;
export const MAX_TAGS = 10;
export const MAX_TAG_LENGTH = 30;

/** Trims a name to something storable; empty string means "no valid name". */
export function normalizeName(name: string | null | undefined): string {
  return (name ?? '').trim().replace(/\s+/g, ' ').slice(0, MAX_NAME_LENGTH);
}

/** Cleans a tag list: trimmed, spaces collapsed, no empties, no duplicates (case-insensitive,
 * first spelling wins), each at most MAX_TAG_LENGTH long, at most MAX_TAGS of them. */
export function normalizeTags(tags: readonly string[] | null | undefined): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const raw of tags ?? []) {
    const tag = String(raw).trim().replace(/\s+/g, ' ').slice(0, MAX_TAG_LENGTH);
    const key = tag.toLowerCase();
    if (!tag || seen.has(key)) continue;
    seen.add(key);
    result.push(tag);
    if (result.length >= MAX_TAGS) break;
  }
  return result;
}
