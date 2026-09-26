"""
rubiks-cube-solver
An optimal 2x2x2 Rubik's cube solver in pure Python (no dependencies).

The cube is modelled as corner permutation + orientation (the classic
cubie model). Scrambles are solved with IDA* search and an admissible
heuristic, so every solution printed is provably optimal (shortest possible).

Run:
    python3 cube.py
"""
import random

# Corner indices: 0=URF 1=UFL 2=ULB 3=UBR 4=DFR 5=DLF 6=DBL 7=DRB
_BASE = {
    "U": ((3, 0, 1, 2, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    "R": ((4, 1, 2, 0, 7, 5, 6, 3), (2, 0, 0, 1, 1, 0, 0, 2)),
    "F": ((1, 5, 2, 3, 0, 4, 6, 7), (1, 2, 0, 0, 2, 1, 0, 0)),
}


def compose(a, b):
    """Return the move that applies a, then b."""
    (acp, aco), (bcp, bco) = a, b
    return (tuple(acp[bcp[i]] for i in range(8)),
            tuple((aco[bcp[i]] + bco[i]) % 3 for i in range(8)))


MOVES = []  # (name, perm, orient)
for face, base in _BASE.items():
    double = compose(base, base)
    inv = compose(double, base)
    MOVES += [(face, *base), (face + "2", *double), (face + "'", *inv)]


def apply_move(state, move):
    (cp, co), (_, mcp, mco) = state, move
    return (tuple(cp[mcp[i]] for i in range(8)),
            tuple((co[mcp[i]] + mco[i]) % 3 for i in range(8)))


SOLVED = (tuple(range(8)), (0,) * 8)


def heuristic(state):
    """Admissible: one move touches at most 4 corners."""
    cp, co = state
    bad = sum(1 for i in range(8) if cp[i] != i or co[i] != 0)
    return (bad + 3) // 4


def _search(state, g, bound, path, prev_face):
    f = g + heuristic(state)
    if f > bound:
        return f
    if state == SOLVED:
        return "FOUND"
    best = float("inf")
    for name, mcp, mco in MOVES:
        if prev_face is not None and name[0] == prev_face:
            continue  # never turn the same face twice in a row
        t = _search(apply_move(state, (None, mcp, mco)), g + 1,
                    bound, path, name[0])
        if t == "FOUND":
            path.append(name)
            return "FOUND"
        best = min(best, t)
    return best


def solve(state):
    """Return an optimal move list, or None if unsolvable (never happens)."""
    bound = heuristic(state)
    while True:
        path = []
        t = _search(state, 0, bound, path, None)
        if t == "FOUND":
            return path[::-1]
        if t == float("inf"):
            return None
        bound = t


def scramble(state, n=8, seed=None):
    rng = random.Random(seed)
    seq, prev = [], None
    for _ in range(n):
        face = rng.choice([f for f in "URF" if f != prev])
        mv = rng.choice([m for m in MOVES if m[0][0] == face])
        seq.append(mv[0])
        state = apply_move(state, mv)
        prev = face
    return state, seq


def main():
    state, seq = scramble(SOLVED, n=8, seed=7)
    print(f"scramble ({len(seq)} moves): {' '.join(seq)}")
    solution = solve(state)
    print(f"solution ({len(solution)} moves): {' '.join(solution)}")
    check = state
    for name in solution:
        check = apply_move(check, next(m for m in MOVES if m[0] == name))
    print("verified:", "SOLVED ✓" if check == SOLVED else "FAILED ✗")


if __name__ == "__main__":
    main()
