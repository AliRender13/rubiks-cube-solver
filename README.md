# rubiks-cube-solver

A Rubik's cube solver in **pure Python** — no dependencies.

## 3x3x3 solver (beginner's method)

```bash
python3 cube3.py demo
```

Runs a deterministic scramble-and-solve demo using the classic beginner's
layer-by-layer method:

1. **White cross** – solve the 4 white edges on U.
2. **White corners** – place the 4 white corners (first layer).
3. **Middle layer edges** – insert the 4 middle edges (second layer).
4. **Yellow cross** – form the yellow cross on D.
5. **Yellow face** – orient the last-layer corners (OLL).
6. **Position last layer** – permute the last layer (PLL).

Moves are grouped by stage with per-stage and total counts, and an ASCII
cube net is printed after every stage.

### How it works

- The cube is a 54-sticker facelet model. Each face turn is a fixed
  permutation generated from 3D rotations, so moves compose exactly like a
  real cube.
- Stage 1 (white cross) uses IDA* search for reliability.
- Stage 2 (white corners) uses explicit beginner insert algorithms.
- Stages 3–6 use explicit algorithms (no search): middle-layer inserts,
  yellow cross (F R U R' U' F'), yellow face (Sune/Anti-Sune), and PLL
  (T-perm for corners, Ua/H/Z-perms for edges).
- `solve_full(scramble)` solves any scramble: pass a list of moves like
  `["R", "U", "R'"]` and get back the full solution. Verified 30/30 on
  random 25-move scrambles.

## 2x2x2 solver (optimal)

An **optimal** 2x2x2 solver — every solution is provably the shortest possible.

```bash
python3 cube.py
```

You'll see a random scramble, then the shortest possible solution that
unscrambles it, verified move-by-move.

### How it works

- The cube uses the classic **cubie model**: 8 corners, each with a position
  (permutation) and a twist (orientation). Every face turn is a fixed
  permutation + orientation table, so moves compose exactly like the real cube.
- Solving uses **IDA\*** (iterative-deepening A\*) with the admissible
  heuristic `ceil(misplaced_corners / 4)` — one turn can fix at most 4
  corners, so the first solution found is guaranteed optimal.
- God's Number for the 2x2 is 11, so even the worst scramble solves in
  at most 11 moves.

## Try changing

- The scramble seed in `cube3.py demo` or `cube.py main()` and race the solver.
- Add move animations or a GUI on top of the facelet model.

Built by [Mohammad Ali](https://github.com/AliRender13).
