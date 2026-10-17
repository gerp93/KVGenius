/**
 * How alike prompts are, for the Prompts page: clusters of near-identical prompts and an
 * "order by similarity" that puts the closest ones side by side. Pure text comparison on the
 * words, so it needs no model and is instant for a few hundred prompts.
 */

/** How strict clustering is: the share of words two prompts have to have in common (0-1). */
export const SIMILARITY = { min: 0.5, max: 0.95, step: 0.05, default: 0.75 } as const;

/** A threshold from storage or a slider: a number within the limits, else the usual one. */
export function cleanSimilarity(value: unknown): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return SIMILARITY.default;
  return Math.min(SIMILARITY.max, Math.max(SIMILARITY.min, Math.round(n / SIMILARITY.step) * SIMILARITY.step));
}

/** The distinct lowercase words of a prompt, in order of first use. Emphasis weights like `(word:1.2)` are not words. */
export function promptWords(prompt: string): string[] {
  const text = prompt.toLowerCase().replace(/:\s*-?\d+(?:\.\d+)?/g, ' ');
  return [...new Set(text.match(/[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu) ?? [])];
}

/** Share of words two prompts have in common (Jaccard): 1 = same words, 0 = nothing shared. */
export function similarity(a: readonly string[], b: readonly string[]): number {
  if (a.length === 0 && b.length === 0) return 1;
  const inB = new Set(b);
  let shared = 0;
  for (const word of a) if (inB.has(word)) shared++;
  return shared / (a.length + b.length - shared);
}

/** The pairwise similarity of every prompt (symmetric, 1 on the diagonal). */
export function similarityMatrix(prompts: readonly string[]): number[][] {
  const words = prompts.map(promptWords);
  const matrix = prompts.map(() => new Array<number>(prompts.length).fill(1));
  for (let i = 0; i < words.length; i++) {
    for (let j = i + 1; j < words.length; j++) matrix[i][j] = matrix[j][i] = similarity(words[i], words[j]);
  }
  return matrix;
}

/**
 * Clusters of prompts, as lists of indexes into the input. Average linkage: two clusters merge while the
 * average similarity between their members is at least `threshold`, so a chain of small changes
 * (A~B, B~C) does not glue together prompts that are far from each other (A and C). Clusters and the
 * indexes inside them come back in input order.
 */
export function clusterPrompts(matrix: readonly (readonly number[])[], threshold: number): number[][] {
  const clusters: number[][] = matrix.map((_, i) => [i]);
  // Average similarity between clusters, kept up to date as they merge (UPGMA).
  const avg = matrix.map((row) => [...row]);
  const alive = new Set(clusters.keys());
  for (;;) {
    let best = -1;
    let bi = -1;
    let bj = -1;
    for (const i of alive) {
      for (const j of alive) {
        if (j <= i) continue;
        if (avg[i][j] > best) {
          best = avg[i][j];
          bi = i;
          bj = j;
        }
      }
    }
    if (bi < 0 || best < threshold) break;
    const si = clusters[bi].length;
    const sj = clusters[bj].length;
    for (const k of alive) {
      if (k === bi || k === bj) continue;
      avg[bi][k] = avg[k][bi] = (si * avg[bi][k] + sj * avg[bj][k]) / (si + sj);
    }
    clusters[bi] = [...clusters[bi], ...clusters[bj]].sort((x, y) => x - y);
    clusters[bj] = [];
    alive.delete(bj);
  }
  return clusters.filter((c) => c.length > 0).sort((x, y) => x[0] - y[0]);
}

/**
 * Orders the given indexes so neighbours are as alike as possible: start at the first one, then always
 * step to the most similar one not yet placed. Ties keep the input order.
 */
export function orderBySimilarity(indexes: readonly number[], matrix: readonly (readonly number[])[]): number[] {
  const left = [...indexes];
  const ordered: number[] = [];
  while (left.length > 0) {
    const last = ordered[ordered.length - 1];
    let pick = 0;
    if (last !== undefined) {
      for (let k = 1; k < left.length; k++) if (matrix[last][left[k]] > matrix[last][left[pick]]) pick = k;
    }
    ordered.push(...left.splice(pick, 1));
  }
  return ordered;
}

/** What a cluster has in common and where its prompts differ: the words not every prompt uses. */
export function varyingWords(prompts: readonly string[]): string[] {
  const lists = prompts.map(promptWords);
  const counts = new Map<string, number>();
  for (const list of lists) for (const word of list) counts.set(word, (counts.get(word) ?? 0) + 1);
  return [...counts.entries()].filter(([, count]) => count < lists.length).map(([word]) => word);
}
