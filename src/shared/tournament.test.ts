import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Pick, TournamentState, answer, answered, currentPair, isDone, questionsLeft, startTournament, undo, winner } from './tournament';

/** A fixed "random" order: the identity, so the first two ids are the first pair. */
const inOrder = () => 0.999999;

const everyone = (s: TournamentState): number[] => [...(s.champion === null ? [] : [s.champion]), ...s.challengers, ...s.out].sort((a, b) => a - b);

/** A small deterministic generator, for property-style checks. */
function lcg(seed: number): () => number {
  let x = seed;
  return () => {
    x = (x * 1664525 + 1013904223) % 4294967296;
    return x / 4294967296;
  };
}

test('with N items there are N-1 questions however they are answered', () => {
  for (const n of [2, 3, 5, 12]) {
    for (const pattern of ['a', 'b', 'ab'] as const) {
      let state = startTournament(Array.from({ length: n }, (_, i) => i + 1), lcg(n));
      let asked = 0;
      while (!isDone(state)) {
        assert.equal(questionsLeft(state), n - 1 - asked, `n=${n} pattern=${pattern}`);
        state = answer(state, pattern === 'ab' ? (asked % 2 === 0 ? 'a' : 'b') : pattern);
        asked++;
      }
      assert.equal(asked, n - 1);
      assert.notEqual(winner(state), null);
      assert.equal(state.out.length, n - 1);
    }
  }
});

test('the pick stays and meets the next item; the other is out', () => {
  let state = startTournament([1, 2, 3, 4], inOrder);
  assert.deepEqual(currentPair(state), [1, 2]);

  state = answer(state, 'a'); // 1 beats 2
  assert.equal(state.champion, 1);
  assert.deepEqual(currentPair(state), [1, 3]);

  state = answer(state, 'b'); // 3 beats 1
  assert.equal(state.champion, 3);
  assert.deepEqual(currentPair(state), [3, 4]);

  state = answer(state, 'a'); // 3 beats 4
  assert.equal(isDone(state), true);
  assert.equal(winner(state), 3);
  assert.deepEqual(state.out, [2, 1, 4]);
});

test('"neither" drops both, and the next two take their place', () => {
  let state = startTournament([1, 2, 3, 4, 5], inOrder);
  state = answer(state, 'neither'); // 1 and 2 both bad
  assert.equal(state.champion, null);
  assert.deepEqual(state.out, [1, 2]);
  assert.deepEqual(currentPair(state), [3, 4]);

  state = answer(state, 'b'); // 4 beats 3
  assert.deepEqual(currentPair(state), [4, 5]);
  state = answer(state, 'a');
  assert.equal(winner(state), 4);
});

test('a lone survivor is the best by default, and dropping everything leaves no winner', () => {
  // 1 and 2 are dropped, leaving 3 with nobody to be compared with.
  let state = startTournament([1, 2, 3], inOrder);
  state = answer(state, 'neither');
  assert.equal(isDone(state), true);
  assert.equal(winner(state), 3);

  // Everything dropped.
  state = startTournament([1, 2, 3, 4], inOrder);
  state = answer(state, 'neither');
  state = answer(state, 'neither');
  assert.equal(isDone(state), true);
  assert.equal(winner(state), null);
  assert.deepEqual(state.out, [1, 2, 3, 4]);
});

test('one item is its own winner at once, and nothing at all has no winner', () => {
  const one = startTournament([7]);
  assert.equal(isDone(one), true);
  assert.equal(winner(one), 7);
  assert.equal(currentPair(one), null);

  const none = startTournament([]);
  assert.equal(isDone(none), true);
  assert.equal(winner(none), null);
});

test('a repeated id is one item', () => {
  const state = startTournament([4, 4, 4, 9]);
  assert.equal(state.total, 2);
  assert.equal(questionsLeft(state), 1);
});

test('undo takes back the last answer, as far back as the start', () => {
  const start = startTournament([1, 2, 3, 4], inOrder);
  const s1 = answer(start, 'a');
  const s2 = answer(s1, 'neither');
  assert.equal(answered(s2), 2);

  const back1 = undo(s2);
  assert.deepEqual({ ...back1, history: [] }, { ...s1, history: [] });
  assert.equal(answered(back1), 1);
  const back2 = undo(back1);
  assert.deepEqual(currentPair(back2), currentPair(start));
  assert.equal(answered(back2), 0);
  assert.equal(undo(back2), back2, 'nothing earlier to go back to');

  // Undoing from the end reopens the last question.
  let done = startTournament([1, 2], inOrder);
  done = answer(done, 'b');
  assert.equal(isDone(done), true);
  assert.deepEqual(currentPair(undo(done)), [1, 2]);
});

test('answering once it is over changes nothing', () => {
  let state = startTournament([1, 2], inOrder);
  state = answer(state, 'a');
  assert.equal(answer(state, 'b'), state);
});

test('nobody is ever lost or duplicated, whatever the answers', () => {
  const rng = lcg(42);
  const choices: Pick[] = ['a', 'b', 'neither'];
  for (let round = 0; round < 200; round++) {
    const n = 1 + Math.floor(rng() * 14);
    const ids = Array.from({ length: n }, (_, i) => i + 1);
    let state = startTournament(ids, rng);
    assert.deepEqual(everyone(state), ids);
    while (!isDone(state)) {
      state = answer(state, choices[Math.floor(rng() * 3)]);
      assert.deepEqual(everyone(state), ids, `round ${round}`);
    }
    // Over: at most one item left standing, and it is the winner.
    assert.equal(state.challengers.length <= (state.champion === null ? 1 : 0), true);
  }
});

test('the order is shuffled, so the first item does not decide the result', () => {
  const ids = Array.from({ length: 20 }, (_, i) => i + 1);
  const firsts = new Set<number>();
  for (let seed = 1; seed <= 30; seed++) firsts.add(currentPair(startTournament(ids, lcg(seed)))![0]);
  assert.ok(firsts.size > 5, 'different seeds start from different items');
});
