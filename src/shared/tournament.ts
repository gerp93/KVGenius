/**
 * "Which is better, A or B?" - the eye-doctor way of picking the best of a set. Two items are shown;
 * the one you pick stays on screen and is shown against the next, until every item has been seen and
 * the one left standing is the best. With N items that is N-1 questions, however you answer.
 *
 * "Neither" drops both (for when they are both bad); the next two then take their place. Everything is
 * plain data, so a step can be undone.
 */

export type Pick = 'a' | 'b' | 'neither';

export interface TournamentState {
  /** The current best, or null while there is none (the start, or right after "neither"). */
  champion: number | null;
  /** Still to be shown, in the order they will come up. */
  challengers: number[];
  /** Dropped so far, in the order they went. */
  out: number[];
  /** How many items there were. */
  total: number;
  /** Earlier states, for undo. */
  history: Omit<TournamentState, 'history'>[];
}

/** A random order, so which item happens to be first never decides the result. */
function shuffled<T>(items: readonly T[], rng: () => number): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/** One survivor with nobody left to be compared with is the best by default. */
function settle(state: Omit<TournamentState, 'history'>): Omit<TournamentState, 'history'> {
  if (state.champion === null && state.challengers.length === 1) {
    return { ...state, champion: state.challengers[0], challengers: [] };
  }
  return state;
}

export function startTournament(ids: readonly number[], rng: () => number = Math.random): TournamentState {
  const unique = [...new Set(ids)];
  return { ...settle({ champion: null, challengers: shuffled(unique, rng), out: [], total: unique.length }), history: [] };
}

/** The two items to show now - [A, B] - or null when it is over. */
export function currentPair(state: TournamentState): [number, number] | null {
  if (state.champion !== null) return state.challengers.length > 0 ? [state.champion, state.challengers[0]] : null;
  return state.challengers.length >= 2 ? [state.challengers[0], state.challengers[1]] : null;
}

export function isDone(state: TournamentState): boolean {
  return currentPair(state) === null;
}

/** The best item once it is over; null if every item was dropped with "neither" (or there were none). */
export function winner(state: TournamentState): number | null {
  return isDone(state) ? state.champion : null;
}

/** How many questions are left at most (fewer if "neither" is used). */
export function questionsLeft(state: TournamentState): number {
  if (state.champion !== null) return state.challengers.length;
  return Math.max(0, state.challengers.length - 1);
}

/** Answers the current question. Does nothing once it is over. */
export function answer(state: TournamentState, choice: Pick): TournamentState {
  const pair = currentPair(state);
  if (!pair) return state;
  const [a, b] = pair;
  // How many challengers the question used up: B only when A is the standing champion.
  const used = state.champion !== null ? 1 : 2;
  const rest = state.challengers.slice(used);

  let champion: number | null;
  let dropped: number[];
  if (choice === 'a') {
    champion = a;
    dropped = [b];
  } else if (choice === 'b') {
    champion = b;
    dropped = [a];
  } else {
    champion = null;
    dropped = [a, b];
  }

  const { history, ...before } = state;
  const next = settle({ champion, challengers: rest, out: [...state.out, ...dropped], total: state.total });
  return { ...next, history: [...history, before] };
}

/** Takes back the last answer. */
export function undo(state: TournamentState): TournamentState {
  if (state.history.length === 0) return state;
  const previous = state.history[state.history.length - 1];
  return { ...previous, history: state.history.slice(0, -1) };
}

/** How many questions have been answered (what undo can take back). */
export function answered(state: TournamentState): number {
  return state.history.length;
}
