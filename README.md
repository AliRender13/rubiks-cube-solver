# rubiks-cube-solver

An **optimal** 2x2x2 Rubik's cube solver in **pure Python** — no dependencies.

## Run it

```bash
python3 cube.py
```

You'll see a random scramble, then the shortest possible solution that
unscrambles it, verified move-by-move.

## How it works

- The cube uses the classic **cubie model**: 8 corners, each with a position
  (permutation) and a twist (orientation). Every face turn is a fixed
  permutation + orientation table, so moves compose exactly like the real cube.
- Solving uses **IDA\*** (iterative-deepening A\*) with the admissible
  heuristic `ceil(misplaced_corners / 4)` — one turn can fix at most 4
  corners, so the first solution found is guaranteed optimal.
- God's Number for the 2x2 is 11, so even the worst scramble solves in
  at most 11 moves.

## Try changing

- The scramble length / seed in `main()` and race the solver.
- Add `B`, `L`, `D` moves (they're redundant on a 2x2 — can you see why?).

Built by [Mohammad Ali](https://github.com/AliRender13).
