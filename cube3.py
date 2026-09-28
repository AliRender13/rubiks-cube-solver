"""
cube3.py - a full 3x3x3 Rubik's cube solver (beginner's layer method), pure Python.

Model: facelet representation (6 faces x 9 stickers). Moves are derived
from 3D rotations of the cube's layers, so they are correct by construction.

Stages (white on U, yellow on D, green on F):
  1. white cross      - daisy method: white edges to D, align, F2/R2/B2/L2
  2. white corners    - knock to D if needed, D-align, orient+insert
  3. middle edges     - second layer edges from U
  4. yellow cross     - last-layer edge orientation
  5. yellow face      - orient last-layer corners
  6. pll-lite         - position last-layer corners, then edges

Run:
    python3 cube3.py          # demo: scramble + staged solve with cube nets
"""
import random

FACES = ('U', 'D', 'F', 'B', 'L', 'R')
FI = {f: i for i, f in enumerate(FACES)}


def _geom(face, idx):
    """3D position + outward normal of a facelet.

    Cube coords x,y,z in {-1,0,1}. Faces are viewed from the outside
    with U up (for the sides); D is viewed from below with F at top.
    """
    r, c = divmod(idx, 3)
    if face == 'U':
        return (c - 1, 1, r - 1), (0, 1, 0)
    if face == 'D':
        return (c - 1, -1, 1 - r), (0, -1, 0)
    if face == 'F':
        return (c - 1, 1 - r, 1), (0, 0, 1)
    if face == 'B':
        return (1 - c, 1 - r, -1), (0, 0, -1)
    if face == 'R':
        return (1, 1 - r, 1 - c), (1, 0, 0)
    if face == 'L':
        return (-1, 1 - r, c - 1), (-1, 0, 0)
    raise ValueError(face)


def _rot(axis, ang):
    """90-degree rotation about an axis; ang=+1 -> +90 deg (right-hand rule)."""
    def f(p):
        x, y, z = p
        if axis == 'x':
            return (x, -z, y) if ang == 1 else (x, z, -y)
        if axis == 'y':
            return (z, y, -x) if ang == 1 else (-z, y, x)
        return (-y, x, z) if ang == 1 else (y, -x, z)  # z
    return f


# face -> (rotation axis, angle, layer coordinate), clockwise seen from outside
_MOVE_DEFS = {
    'U': ('y', -1, 1), 'D': ('y', 1, -1),
    'F': ('z', -1, 1), 'B': ('z', 1, -1),
    'R': ('x', -1, 1), 'L': ('x', 1, -1),
}


def _build_perm(face):
    axis, ang, layer = _MOVE_DEFS[face]
    rot = _rot(axis, ang)
    ax = {'x': 0, 'y': 1, 'z': 2}[axis]
    geoms = {(f, i): _geom(f, i) for f in FACES for i in range(9)}
    inv = {v: k for k, v in geoms.items()}
    perm = [0] * 54  # new[d] = old[perm[d]]
    for (f, i), (pos, n) in geoms.items():
        s = FI[f] * 9 + i
        if pos[ax] == layer:
            pos, n = rot(pos), rot(n)
        f2, i2 = inv[(pos, n)]
        perm[FI[f2] * 9 + i2] = s
    return perm


def _compose(p, q):
    """Permutation applying p, then q: new[i] = old[p[q[i]]]."""
    return [p[q[i]] for i in range(54)]


_PERMS = {}
for _face in FACES:
    _base = _build_perm(_face)
    _dbl = _compose(_base, _base)
    _inv = _compose(_dbl, _base)
    _PERMS[_face] = _base
    _PERMS[_face + '2'] = _dbl
    _PERMS[_face + "'"] = _inv
del _face, _base, _dbl, _inv


def apply(state, move):
    """Return the state after one move (move like "R", "U'", "F2")."""
    p = _PERMS[move]
    return tuple(state[p[i]] for i in range(54))


def apply_seq(state, moves):
    for m in moves:
        state = apply(state, m)
    return state


SOLVED = tuple('W' * 9 + 'Y' * 9 + 'G' * 9 + 'B' * 9 + 'O' * 9 + 'R' * 9)


def get(state, face, idx):
    return state[FI[face] * 9 + idx]


def scramble(state, n=25, seed=None):
    rng = random.Random(seed)
    faces = list(FACES)
    seq, prev = [], None
    for _ in range(n):
        face = rng.choice([f for f in faces if f != prev])
        mv = face + rng.choice(['', '2', "'"])
        seq.append(mv)
        state = apply(state, mv)
        prev = face
    return state, seq


# ---------------------------------------------------------------------------
# ASCII visualizer: unfolded cube net
# ---------------------------------------------------------------------------

def net_str(state):
    g = lambda f, i: get(state, f, i)  # noqa: E731
    L = []
    for r in range(3):
        L.append('        ' + ' '.join(g('U', r * 3 + c) for c in range(3)))
    for r in range(3):
        row = []
        for f in ('L', 'F', 'R', 'B'):
            row.append(' '.join(g(f, r * 3 + c) for c in range(3)))
        L.append('  '.join(row))
    for r in range(3):
        L.append('        ' + ' '.join(g('D', r * 3 + c) for c in range(3)))
    return '\n'.join(L)


# ---------------------------------------------------------------------------
# Piece helpers
# ---------------------------------------------------------------------------

EDGES = {
    'UF': (('U', 7), ('F', 1)), 'UR': (('U', 5), ('R', 1)),
    'UB': (('U', 1), ('B', 1)), 'UL': (('U', 3), ('L', 1)),
    'DF': (('D', 1), ('F', 7)), 'DR': (('D', 5), ('R', 7)),
    'DB': (('D', 7), ('B', 7)), 'DL': (('D', 3), ('L', 7)),
    'FR': (('F', 5), ('R', 3)), 'FL': (('F', 3), ('L', 5)),
    'BR': (('B', 3), ('R', 5)), 'BL': (('B', 5), ('L', 3)),
}
CORNERS = {
    'UFR': (('U', 8), ('F', 2), ('R', 0)),
    'UFL': (('U', 6), ('F', 0), ('L', 2)),
    'UBL': (('U', 0), ('B', 2), ('L', 0)),
    'UBR': (('U', 2), ('B', 0), ('R', 2)),
    'DFR': (('D', 2), ('F', 8), ('R', 6)),
    'DFL': (('D', 0), ('F', 6), ('L', 8)),
    'DBL': (('D', 6), ('B', 8), ('L', 6)),
    'DBR': (('D', 8), ('B', 6), ('R', 8)),
}
D_POS = ('DF', 'DR', 'DB', 'DL')
U_POS = ('UF', 'UR', 'UB', 'UL')
MID_POS = ('FR', 'FL', 'BR', 'BL')
D_CORNERS = ('DFR', 'DFL', 'DBL', 'DBR')
U_CORNERS = ('UFR', 'UFL', 'UBL', 'UBR')


def edge_colors(state, pos):
    (f1, i1), (f2, i2) = EDGES[pos]
    return (get(state, f1, i1), get(state, f2, i2))


def find_edge(state, ca, cb):
    """Locate the edge with colors {ca, cb}; return (pos, idx_of_ca)."""
    for pos in EDGES:
        cols = edge_colors(state, pos)
        if set(cols) == {ca, cb}:
            return pos, cols.index(ca)
    raise AssertionError('edge not found')


def corner_colors(state, pos):
    return tuple(get(state, f, i) for f, i in CORNERS[pos])


def find_corner(state, ca, cb, cc):
    """Locate the corner with colors {ca, cb, cc}; return (pos, idx_of_ca)."""
    want = {ca, cb, cc}
    for pos in CORNERS:
        cols = corner_colors(state, pos)
        if set(cols) == want:
            return pos, cols.index(ca)
    raise AssertionError('corner not found')


# ---------------------------------------------------------------------------
# Staged beginner's-method solver
# ---------------------------------------------------------------------------
# Each stage is solved with IDA* (iterative deepening A*) using a stage-specific
# goal test and heuristic. The stages follow the beginner's layer-by-layer
# method: white cross -> white corners -> middle edges -> yellow cross ->
# yellow face -> PLL (position last layer).

_ALL_MOVES = [m for f in "URFDLB" for m in (f, f + "'", f + "2")]


def _ida_search(state, g, bound, path, prev_face, goal_test, heuristic, moves):
    f = g + heuristic(state)
    if f > bound:
        return f
    if goal_test(state):
        return "FOUND"
    best = float("inf")
    for m in moves:
        if prev_face is not None and m[0] == prev_face:
            continue
        ns = apply(state, m)
        t = _ida_search(ns, g + 1, bound, path, m[0], goal_test, heuristic, moves)
        if t == "FOUND":
            path.append(m)
            return "FOUND"
        if t < best:
            best = t
    return best


def ida_solve(state, goal_test, heuristic, moves=None, max_bound=20):
    """IDA* search. Returns move list or None."""
    if moves is None:
        moves = _ALL_MOVES
    bound = heuristic(state)
    while bound <= max_bound:
        path = []
        t = _ida_search(state, 0, bound, path, None, goal_test, heuristic, moves)
        if t == "FOUND":
            return path[::-1]
        if t == float("inf"):
            return None
        bound = t
    return None


# Stage goals and heuristics
def _white_cross_goal(st):
    return (edge_colors(st, 'UF') == ('W', 'G') and
            edge_colors(st, 'UR') == ('W', 'R') and
            edge_colors(st, 'UB') == ('W', 'B') and
            edge_colors(st, 'UL') == ('W', 'O'))


def _white_cross_h(st):
    bad = sum(1 for pos, want in (('UF', ('W', 'G')), ('UR', ('W', 'R')),
                                  ('UB', ('W', 'B')), ('UL', ('W', 'O')))
              if edge_colors(st, pos) != want)
    return (bad + 1) // 2


def _corners_goal(st):
    return (_white_cross_goal(st) and
            corner_colors(st, 'UFR') == ('W', 'G', 'R') and
            corner_colors(st, 'UFL') == ('W', 'G', 'O') and
            corner_colors(st, 'UBL') == ('W', 'B', 'O') and
            corner_colors(st, 'UBR') == ('W', 'B', 'R'))


def _corners_h(st):
    bad = sum(1 for pos, want in (('UFR', ('W', 'G', 'R')),
                                  ('UFL', ('W', 'G', 'O')),
                                  ('UBL', ('W', 'B', 'O')),
                                  ('UBR', ('W', 'B', 'R')))
              if corner_colors(st, pos) != want)
    return bad


def _f2l_goal(st):
    return (_corners_goal(st) and
            edge_colors(st, 'FR') == ('G', 'R') and
            edge_colors(st, 'FL') == ('O', 'G') and
            edge_colors(st, 'BR') == ('R', 'B') and
            edge_colors(st, 'BL') == ('B', 'O'))


def _f2l_h(st):
    bad = sum(1 for pos, want in (('FR', ('G', 'R')), ('FL', ('O', 'G')),
                                  ('BR', ('R', 'B')), ('BL', ('B', 'O')))
              if edge_colors(st, pos) != want)
    return bad


def _yellow_cross_goal(st):
    return (_f2l_goal(st) and
            get(st, 'D', 1) == 'Y' and get(st, 'D', 3) == 'Y' and
            get(st, 'D', 5) == 'Y' and get(st, 'D', 7) == 'Y')


def _yellow_cross_h(st):
    bad = sum(1 for i in (1, 3, 5, 7) if get(st, 'D', i) != 'Y')
    return (bad + 3) // 4


def _yellow_face_goal(st):
    return (_f2l_goal(st) and
            all(get(st, 'D', i) == 'Y' for i in range(9)))


def _yellow_face_h(st):
    bad = sum(1 for i in range(9) if get(st, 'D', i) != 'Y')
    return (bad + 3) // 4


def _solved_goal(st):
    return st == SOLVED


def _solved_h(st):
    # count misplaced stickers / 8 (a move affects many stickers)
    bad = sum(1 for i in range(54) if st[i] != SOLVED[i])
    return (bad + 7) // 8


STAGES = [
    ("White cross", _white_cross_goal, _white_cross_h),
    ("White corners", _corners_goal, _corners_h),
    ("Middle layer edges", _f2l_goal, _f2l_h),
    ("Yellow cross", _yellow_cross_goal, _yellow_cross_h),
    ("Yellow face", _yellow_face_goal, _yellow_face_h),
    ("Position last layer", _solved_goal, _solved_h),
]


def _bfs_limited(state, goal_test, preserve_test, max_depth=9, max_states=300000):
    """BFS with state limit. Returns move list or None."""
    from collections import deque
    seen = {state}
    q = deque([(state, [])])
    while q:
        st, path = q.popleft()
        if goal_test(st) and preserve_test(st):
            return path
        if len(path) >= max_depth:
            continue
        for m in _ALL_MOVES:
            if path and m[0] == path[-1][0]:
                continue
            ns = apply(st, m)
            if ns not in seen:
                if len(seen) >= max_states:
                    return None
                seen.add(ns)
                q.append((ns, path + [m]))
    return None


def solve_beginner(state, verbose=False):
    """Solve using the beginner's layer method.

    Returns (stages, total_moves, final_state). Each stage dict has
    name, moves, count, state, net.
    """
    from collections import deque
    stages = []
    st = state

    # --- Stage 1: White cross (BFS, piece by piece) ---
    smoves = []
    done_edges = []
    for pos, want in [('UF', ('W', 'G')), ('UR', ('W', 'R')),
                      ('UB', ('W', 'B')), ('UL', ('W', 'O'))]:
        def goal(s, p=pos, w=want):
            return edge_colors(s, p) == w
        def preserve(s):
            for pp, ww in done_edges:
                if edge_colors(s, pp) != ww:
                    return False
            return True
        mv = _bfs_limited(st, goal, preserve, max_depth=9)
        if mv is None:
            raise RuntimeError(f"White cross failed at {pos}")
        st = apply_seq(st, mv)
        smoves.extend(mv)
        done_edges.append((pos, want))
    stages.append({"name": "White cross", "moves": smoves,
                   "count": len(smoves), "state": st, "net": net_str(st)})
    if verbose:
        print(f"White cross: {len(smoves)} moves")

    # --- Stage 2: White corners (explicit algorithms) ---
    smoves = []
    INSERT = {
        'UFR': ('DFR', {'R': ["R'", "D'", "R"], 'F': ["F", "D", "F'"],
                        'D': ["R'", "F'", "R", "F", "D", "R"]}),
        'UFL': ('DFL', {'L': ["L", "D", "L'"], 'F': ["F'", "D'", "F"],
                        'D': ["L", "F", "L'", "F'", "D'", "L'"]}),
        'UBL': ('DBL', {'L': ["L'", "D'", "L"], 'B': ["B", "D", "B'"],
                        'D': ["L'", "B'", "L", "B", "D", "L"]}),
        'UBR': ('DBR', {'R': ["R", "D", "R'"], 'B': ["B'", "D'", "B"],
                        'D': ["R", "B", "R'", "B'", "D'", "R'"]}),
    }
    KNOCK = {'UFR': ('DFL', ["R'", "D'", "R"]),
             'UFL': ('DFR', ["L", "D", "L'"]),
             'UBL': ('DBR', ["L'", "D'", "L"]),
             'UBR': ('DBL', ["R", "D", "R'"])}
    SLOT_COLORS = {'UFR': ('W', 'G', 'R'), 'UFL': ('W', 'G', 'O'),
                   'UBL': ('W', 'B', 'O'), 'UBR': ('W', 'B', 'R')}
    D_CYCLE = ['DFR', 'DBR', 'DBL', 'DFL']
    def d_align(frm, to):
        i = D_CYCLE.index(frm)
        j = D_CYCLE.index(to)
        k = (j - i) % 4
        return [[], ["D"], ["D", "D"], ["D'"]][k]
    for slot in ['UFR', 'UFL', 'UBL', 'UBR']:
        target = SLOT_COLORS[slot]
        d_pos, algs = INSERT[slot]
        ca, cb, cc = target
        for _ in range(12):
            pos, _ = find_corner(st, ca, cb, cc)
            if pos == slot and corner_colors(st, pos) == target:
                break
            if pos in KNOCK:
                kd_pos, kd_moves = KNOCK[pos]
                st = apply_seq(st, kd_moves)
                smoves.extend(kd_moves)
                pos = kd_pos
            if pos != d_pos:
                da = d_align(pos, d_pos)
                st = apply_seq(st, da)
                smoves.extend(da)
            cols = corner_colors(st, d_pos)
            wi = cols.index('W')
            fm = {'DFR': ['D', 'F', 'R'], 'DFL': ['D', 'F', 'L'],
                  'DBL': ['D', 'B', 'L'], 'DBR': ['D', 'B', 'R']}[d_pos]
            wf = fm[wi]
            alg = algs[wf]
            st = apply_seq(st, alg)
            smoves.extend(alg)
        else:
            raise RuntimeError(f"White corners failed at {slot}")
    stages.append({"name": "White corners", "moves": smoves,
                   "count": len(smoves), "state": st, "net": net_str(st)})
    if verbose:
        print(f"White corners: {len(smoves)} moves")

    # --- Stage 3: Middle edges (BFS, piece by piece) ---
    smoves = []
    done_mid = []
    # preserve function: white cross + corners + done_mid
    def mid_preserve(s):
        if not (edge_colors(s, 'UF') == ('W', 'G') and
                corner_colors(s, 'UFR') == ('W', 'G', 'R')):
            return False
        for pp, ww in done_mid:
            if edge_colors(s, pp) != ww:
                return False
        return True
    for pos, want in [('FR', ('G', 'R')), ('FL', ('O', 'G')),
                      ('BR', ('R', 'B')), ('BL', ('B', 'O'))]:
        def goal(s, p=pos, w=want):
            return edge_colors(s, p) == w
        mv = _bfs_limited(st, goal, mid_preserve, max_depth=12, max_states=500000)
        if mv is None:
            raise RuntimeError(f"Middle edges failed at {pos}")
        st = apply_seq(st, mv)
        smoves.extend(mv)
        done_mid.append((pos, want))
    stages.append({"name": "Middle layer edges", "moves": smoves,
                   "count": len(smoves), "state": st, "net": net_str(st)})
    if verbose:
        print(f"Middle layer edges: {len(smoves)} moves")

    def f2l_preserve(s):
        return (edge_colors(s, 'UF') == ('W', 'G') and
                edge_colors(s, 'UR') == ('W', 'R') and
                edge_colors(s, 'UB') == ('W', 'B') and
                edge_colors(s, 'UL') == ('W', 'O') and
                corner_colors(s, 'UFR') == ('W', 'G', 'R') and
                corner_colors(s, 'UFL') == ('W', 'G', 'O') and
                corner_colors(s, 'UBL') == ('W', 'B', 'O') and
                corner_colors(s, 'UBR') == ('W', 'B', 'R') and
                edge_colors(s, 'FR') == ('G', 'R') and
                edge_colors(s, 'FL') == ('O', 'G') and
                edge_colors(s, 'BR') == ('R', 'B') and
                edge_colors(s, 'BL') == ('B', 'O'))

    # --- Stages 4-6: Solve rest via BFS, split by goals ---
    # After F2L, solve the last layer. We'll do a single BFS to SOLVED,
    # then split the moves by when yellow cross / face goals are met.
    def rest_goal(s):
        return s == SOLVED
    # For the final solve, we preserve F2L (stages 1-3)
    mv = _bfs_limited(st, rest_goal, f2l_preserve, max_depth=14, max_states=500000)
    if mv is None:
        raise RuntimeError("Last layer failed")
    # Split mv into stages 4,5,6 by goals
    s4_moves, s5_moves, s6_moves = [], [], []
    cur = st
    # Stage 4: until yellow cross (all D edges Y)
    i = 0
    while i < len(mv):
        cur = apply(cur, mv[i])
        s4_moves.append(mv[i])
        i += 1
        dy = [get(cur, 'D', j) for j in (1, 3, 5, 7)]
        if all(c == 'Y' for c in dy):
            break
    # Stage 5: until yellow face (all D Y)
    while i < len(mv):
        cur = apply(cur, mv[i])
        s5_moves.append(mv[i])
        i += 1
        if all(get(cur, 'D', j) == 'Y' for j in range(9)):
            break
    # Stage 6: rest
    while i < len(mv):
        cur = apply(cur, mv[i])
        s6_moves.append(mv[i])
        i += 1
    st = cur
    stages.append({"name": "Yellow cross", "moves": s4_moves,
                   "count": len(s4_moves), "state": apply_seq(stages[-1]["state"], s4_moves),
                   "net": net_str(apply_seq(stages[-1]["state"], s4_moves))})
    stages.append({"name": "Yellow face", "moves": s5_moves,
                   "count": len(s5_moves),
                   "state": apply_seq(stages[-1]["state"], s5_moves),
                   "net": net_str(apply_seq(stages[-1]["state"], s5_moves))})
    stages.append({"name": "Position last layer", "moves": s6_moves,
                   "count": len(s6_moves), "state": st, "net": net_str(st)})
    if verbose:
        print(f"Yellow cross: {len(s4_moves)} moves")
        print(f"Yellow face: {len(s5_moves)} moves")
        print(f"Position last layer: {len(s6_moves)} moves")

    total = sum(s["count"] for s in stages)
    return stages, total, st


def demo(seed=7, n=25):
    """Deterministic scramble-and-solve demo."""
    st, seq = scramble(SOLVED, n, seed=seed)
    print(f"Scramble ({len(seq)} moves): {' '.join(seq)}\n")
    print("Scrambled cube:")
    print(net_str(st))
    print()
    try:
        stages, total, final = solve_beginner(st, verbose=True)
    except RuntimeError as e:
        print(f"\nSolver stopped: {e}")
        print("Partial solution shown above.")
        return
    print()
    for s in stages:
        print(f"=== {s['name']} ({s['count']} moves) ===")
        if s['moves']:
            print(' '.join(s['moves']))
        print(s['net'])
        print()
    print(f"Total: {total} moves")
    print("Solved:", final == SOLVED)


# ---------------------------------------------------------------------------
# Full 3x3 beginner solver: solve_full(scramble)
#
# Stages (explicit algorithms, no BFS/IDA* for stages 3-6):
#   1. White cross (IDA* - reliable)
#   2. White corners (explicit insert algorithms)
#   3. Middle edges (explicit 3-move inserts, white-D frame)
#   4. Yellow cross (F R U R' U' F')
#   5. Yellow face (Sune/Anti-Sune)
#   6. PLL (T-perm for corners, Ua/H/Z for edges)
# ---------------------------------------------------------------------------

def _build_x2_perm():
    """Permutation for whole-cube X2 rotation (white-U -> white-D)."""
    geoms = {(f, i): _geom(f, i) for f in FACES for i in range(9)}
    inv = {v: k for k, v in geoms.items()}
    perm = [0] * 54
    for (f, i), (pos, n) in geoms.items():
        x, y, z = pos
        nx, ny, nz = n
        f2, i2 = inv[((x, -y, -z), (nx, -ny, -nz))]
        perm[FI[f2] * 9 + i2] = FI[f] * 9 + i
    return perm

_X2 = _build_x2_perm()

def _rot_x2(state):
    """Apply whole-cube X2 rotation."""
    return tuple(state[_X2[i]] for i in range(54))

# Map moves from white-D frame back to white-U frame: U<->D, F<->B, R->R, L->L
_X2MAP = {}
for _m in ["U","U'","U2","D","D'","D2","F","F'","F2","B","B'","B2","R","R'","R2","L","L'","L2"]:
    _f = _m[0]
    _suf = _m[1:]
    _nf = {'U':'D','D':'U','F':'B','B':'F','R':'R','L':'L'}[_f]
    _X2MAP[_m] = _nf + _suf
del _m, _f, _suf, _nf

def _map_x2(moves):
    """Map a move list from white-D frame to white-U frame."""
    return [_X2MAP[m] for m in moves]

# Explicit algorithms (white-D frame for stages 3-6)
_YCROSS = ["F", "R", "U", "R'", "U'", "F'"]
_SUNE = ["R", "U", "R'", "U", "R", "U2", "R'"]
_ANTISUNE = ["R", "U2", "R'", "U'", "R", "U'", "R'"]
_UA = ["R", "U'", "R", "U", "R", "U", "R", "U'", "R'", "U'", "R2"]
_H_PERM = ["R2","U2","R","U2","R2","U2","R2","U2","R","U2","R2"]
_Z_PERM = ["R", "U'", "R", "U", "R", "U", "R", "U'", "R'", "U'", "R2",
           "U'",
           "R", "U'", "R", "U", "R", "U", "R", "U'", "R'", "U'", "R2",
           "U"]
_T_PERM = ["R","U","R'","U'","R'","F","R2","U'","R'","U'","R","U","R'","F'"]
# Middle edge inserts (white-D frame)
_S3R = ["U", "R", "U'", "R'", "U'", "F'", "U", "F"]      # FR slot, from UF
_S3L = ["U'", "L'", "U", "L", "U", "F", "U'", "F'"]      # FL slot, from UF
_S3BR = ["U'", "R'", "U", "R", "U", "B", "U'", "B'"]     # BR slot, from UB
_S3BL = ["U", "L", "U'", "L'", "U'", "B'", "U", "B"]     # BL slot, from UB

def _rep(alg, n):
    out = []
    for _ in range(n):
        out.extend(alg)
    return out

def _dintact(st):
    return all(get(st, 'D', i) == 'W' for i in range(9))

def _ualign_to(st, edge, target):
    """Rotate U to bring edge to target. Returns (state, moves)."""
    for mv in [[], ["U"], ["U2"], ["U'"]]:
        s = apply_seq(st, mv)
        pos, _ = find_edge(s, *edge)
        if pos == target:
            return s, mv
    return st, []

def _solve_stage3(st):
    """Middle edges in white-D frame. Returns (state, moves)."""
    moves = []
    # (slot, (c1,c2), alg, setup_pos, side_face, side_color)
    # side_face is 'F' or 'B'; side_color is the color that should be on side_face at setup.
    slots = [
        ('FR', ('B','R'), _S3R, 'UF', 'F', 'B'),
        ('FL', ('B','O'), _S3L, 'UF', 'F', 'B'),
        ('BR', ('G','R'), _S3BR, 'UB', 'B', 'G'),
        ('BL', ('G','O'), _S3BL, 'UB', 'B', 'G'),
    ]
    # Map slot -> alg for knockouts
    alg_for_slot = {'FR': _S3R, 'FL': _S3L, 'BR': _S3BR, 'BL': _S3BL}

    for slot, (c1, c2), alg, setup_pos, side_face, side_color in slots:
        for attempt in range(8):
            # Check if slot correct
            cols = edge_colors(st, slot)
            # slot orientation: for FR, edge_colors returns (F,R). Want (c1,c2) = (B,R).
            # For BR, edge_colors returns (B,R). Want (G,R). etc.
            want = (c1, c2)
            # Need to check the orientation matches. edge_colors for:
            # FR: (F,R), FL: (F,L), BR: (B,R), BL: (B,L)
            if cols == want and _dintact(st):
                break
            pos, _ = find_edge(st, c1, c2)
            if pos == slot:
                # In slot but flipped (or wrong orientation). Knock out.
                st = apply_seq(st, alg)
                moves.extend(alg)
            elif pos in alg_for_slot:
                # In wrong middle slot. Knock out with that slot's alg.
                st = apply_seq(st, alg_for_slot[pos])
                moves.extend(alg_for_slot[pos])
            else:
                # In U layer. Align to setup_pos.
                st, mv = _ualign_to(st, (c1, c2), setup_pos)
                moves.extend(mv)
                # Check orientation
                cols_u = edge_colors(st, setup_pos)
                # setup_pos UF: (U,F). setup_pos UB: (U,B).
                # Want side_face sticker == side_color.
                # For UF: cols_u = (U,F). side_face='F' -> cols_u[1] should be side_color.
                # For UB: cols_u = (U,B). side_face='B' -> cols_u[1] should be side_color.
                if cols_u[1] == side_color:
                    # Good orientation. Apply alg.
                    st = apply_seq(st, alg)
                    moves.extend(alg)
                else:
                    # Bad orientation. Do 3-alg procedure: alg, alg, ualign, alg.
                    st = apply_seq(st, alg)
                    moves.extend(alg)
                    st = apply_seq(st, alg)
                    moves.extend(alg)
                    st, mv = _ualign_to(st, (c1, c2), setup_pos)
                    moves.extend(mv)
                    st = apply_seq(st, alg)
                    moves.extend(alg)
        else:
            raise RuntimeError(f"Stage 3 failed at {slot}")
        # Verify
        cols = edge_colors(st, slot)
        if cols != want:
            raise RuntimeError(f"Stage 3 verify failed at {slot}: {cols} != {want}")
    return st, moves

def _yellow_cross_done(st):
    return (get(st,'U',4)=='Y' and all(edge_colors(st,p)[0]=='Y' for p in ['UF','UR','UB','UL']))

def _solve_stage4(st):
    moves = []
    # Yellow cross: proper pattern handling
    for _ in range(6):
        if _yellow_cross_done(st):
            break
        # Count oriented edges
        oriented = [p for p in ['UF','UR','UB','UL'] if edge_colors(st,p)[0]=='Y']
        n = len(oriented)
        if n == 0:
            # Dot: apply _YCROSS
            st = apply_seq(st, _YCROSS)
            moves.extend(_YCROSS)
        elif n == 2:
            # L or line. U-align and apply.
            # Try all U rotations to find one that makes progress.
            # For L: want the two oriented at UB+UL (back-left). For line: horizontal.
            # Simplest: try each U, apply _YCROSS, see if we get closer.
            # Actually, just apply _YCROSS; if it's L, we need correct alignment.
            # Let's detect: if adjacent (L) vs opposite (line).
            poss = set(oriented)
            is_line = (poss == {'UF','UB'} or poss == {'UR','UL'})
            if is_line:
                # Line: make it horizontal (UF+UB is vertical? Actually UF-UB is vertical line through center.
                # For F R U R' U' F', the line should be horizontal (left-right).
                # If line is UF+UB (vertical), U-rotate to make it UR+UL (horizontal).
                if poss == {'UF','UB'}:
                    st = apply_seq(st, ["U"])
                    moves.append("U")
            else:
                # L shape: put the corner of the L at UL (i.e., oriented at UB and UL).
                # Find U rotation that puts oriented at UB and UL.
                for mv in [[], ["U"], ["U2"], ["U'"]]:
                    s_test = apply_seq(st, mv)
                    o = set(p for p in ['UF','UR','UB','UL'] if edge_colors(s_test,p)[0]=='Y')
                    if o == {'UB','UL'}:
                        st = s_test
                        moves.extend(mv)
                        break
            st = apply_seq(st, _YCROSS)
            moves.extend(_YCROSS)
        else:
            # Shouldn't happen (n=4 is done, n=1,3 impossible for edges)
            st = apply_seq(st, _YCROSS)
            moves.extend(_YCROSS)
    if not _yellow_cross_done(st):
        raise RuntimeError("Stage 4 failed")
    return st, moves

def _yellow_face_done(st):
    return all(get(st,'U',i)=='Y' for i in range(9))

def _solve_stage5(st):
    moves = []
    # Yellow face: try 1-alg then 2-alg combos of (U-align + Sune/AntiSune)
    # This is exhaustive try, not BFS.
    UOPTS = [[], ["U"], ["U2"], ["U'"]]
    ALGS = [_SUNE, _ANTISUNE]
    for _ in range(5):  # up to 5 rounds (shouldn't need more than 1)
        if _yellow_face_done(st):
            break
        found = False
        # Try 1 alg
        for u in UOPTS:
            for alg in ALGS:
                s_test = apply_seq(st, u + alg)
                if _yellow_face_done(s_test) and _dintact(s_test):
                    st = s_test
                    moves.extend(u + alg)
                    found = True
                    break
            if found:
                break
        if found:
            continue
        # Try 2 algs
        for u1 in UOPTS:
            for alg1 in ALGS:
                for u2 in UOPTS:
                    for alg2 in ALGS:
                        s_test = apply_seq(st, u1 + alg1 + u2 + alg2)
                        if _yellow_face_done(s_test) and _dintact(s_test):
                            st = s_test
                            moves.extend(u1 + alg1 + u2 + alg2)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise RuntimeError("Stage 5 failed: no 1-2 alg combo works")
    if not _yellow_face_done(st):
        raise RuntimeError("Stage 5 failed")
    return st, moves


def _solve_stage6(st):
    """PLL in white-D frame. Returns (state, moves)."""
    moves = []
    S2_local = _rot_x2(SOLVED)
    UOPTS = [[], ["U"], ["U2"], ["U'"]]
    def corners_solved(s):
        return all(corner_colors(s, slot) == want
                   for slot, want in [('UFR',('Y','B','R')), ('UFL',('Y','B','O')),
                                      ('UBL',('Y','G','O')), ('UBR',('Y','G','R'))])
    # Phase 1: corners with T-perm
    found_c = False
    for u1 in UOPTS:
        for t1 in [0, 1]:
            for u2 in UOPTS:
                for t2 in [0, 1]:
                    seq_c = u1 + (_T_PERM if t1 else []) + u2 + (_T_PERM if t2 else [])
                    s_c = apply_seq(st, seq_c)
                    if corners_solved(s_c):
                        st = s_c
                        moves.extend(seq_c)
                        found_c = True
                        break
                if found_c: break
            if found_c: break
        if found_c: break
    if not found_c:
        raise RuntimeError("Stage 6 corners failed")
    # Phase 2: edges with conjugates (preserve corners)
    edge_algs = [_rep(_UA, m) for m in range(3)] + [_H_PERM, _Z_PERM]
    found_e = False
    for k in range(4):
        uk = ["U"] * k
        uk_inv = ["U"] * ((4 - k) % 4)
        for ealg in edge_algs:
            seq_e = uk + ealg + uk_inv
            s_e = apply_seq(st, seq_e)
            if s_e == S2_local:
                st = s_e
                moves.extend(seq_e)
                found_e = True
                break
        if found_e:
            break
    if not found_e:
        raise RuntimeError("Stage 6 edges failed")
    return st, moves

def solve_full(scramble):
    """Solve a 3x3 cube from a scramble.

    Args:
        scramble: list of moves (e.g. ["R", "U", "R'"]) or a scrambled state tuple.

    Returns:
        list of moves that solves the cube.

    Stages:
      1. White cross (IDA*)
      2. White corners (explicit)
      3. Middle edges (explicit, white-D frame)
      4. Yellow cross (explicit)
      5. Yellow face (explicit)
      6. PLL (explicit)
    """
    # Get scrambled state
    if isinstance(scramble, (list, tuple)) and scramble and isinstance(scramble[0], str):
        # List of moves
        st = apply_seq(SOLVED, scramble)
    else:
        # Already a state
        st = scramble

    m12 = []
    # Stage 1: white cross via IDA*
    mv1 = ida_solve(st, _white_cross_goal, _white_cross_h, max_bound=12)
    if mv1 is None:
        raise RuntimeError("White cross failed")
    st = apply_seq(st, mv1)
    m12.extend(mv1)

    # Stage 2: white corners (explicit)
    INSERT = {
        'UFR': ('DFR', {'R': ["R'", "D'", "R"], 'F': ["F", "D", "F'"],
                        'D': ["R'", "F'", "R", "F", "D", "R"]}),
        'UFL': ('DFL', {'L': ["L", "D", "L'"], 'F': ["F'", "D'", "F"],
                        'D': ["L", "F", "L'", "F'", "D'", "L'"]}),
        'UBL': ('DBL', {'L': ["L'", "D'", "L"], 'B': ["B", "D", "B'"],
                        'D': ["L'", "B'", "L", "B", "D", "L"]}),
        'UBR': ('DBR', {'R': ["R", "D", "R'"], 'B': ["B'", "D'", "B"],
                        'D': ["R", "B", "R'", "B'", "D'", "R'"]}),
    }
    KNOCK = {'UFR': ('DFL', ["R'", "D'", "R"]),
             'UFL': ('DFR', ["L", "D", "L'"]),
             'UBL': ('DBR', ["L'", "D'", "L"]),
             'UBR': ('DBL', ["R", "D", "R'"])}
    SLOT_COLORS = {'UFR': ('W', 'G', 'R'), 'UFL': ('W', 'G', 'O'),
                   'UBL': ('W', 'B', 'O'), 'UBR': ('W', 'B', 'R')}
    D_CYCLE = ['DFR', 'DBR', 'DBL', 'DFL']
    def d_align(frm, to):
        i = D_CYCLE.index(frm)
        j = D_CYCLE.index(to)
        k = (j - i) % 4
        return [[], ["D"], ["D", "D"], ["D'"]][k]
    for slot in ['UFR', 'UFL', 'UBL', 'UBR']:
        target = SLOT_COLORS[slot]
        d_pos, algs = INSERT[slot]
        ca, cb, cc = target
        for _ in range(12):
            pos, _ = find_corner(st, ca, cb, cc)
            if pos == slot and corner_colors(st, pos) == target:
                break
            if pos in KNOCK:
                kd_pos, kd_moves = KNOCK[pos]
                st = apply_seq(st, kd_moves)
                m12.extend(kd_moves)
                pos = kd_pos
            if pos != d_pos:
                da = d_align(pos, d_pos)
                st = apply_seq(st, da)
                m12.extend(da)
            cols = corner_colors(st, d_pos)
            wi = cols.index('W')
            fm = {'DFR': ['D', 'F', 'R'], 'DFL': ['D', 'F', 'L'],
                  'DBL': ['D', 'B', 'L'], 'DBR': ['D', 'B', 'R']}[d_pos]
            wf = fm[wi]
            alg = algs[wf]
            st = apply_seq(st, alg)
            m12.extend(alg)
        else:
            raise RuntimeError(f"White corners failed at {slot}")

    # X2: white-U -> white-D
    st = _rot_x2(st)

    # Stages 3-6 in white-D frame
    st, m3 = _solve_stage3(st)
    st, m4 = _solve_stage4(st)
    st, m5 = _solve_stage5(st)
    st, m6 = _solve_stage6(st)

    # Map 3-6 moves back to original frame
    m3456 = _map_x2(m3 + m4 + m5 + m6)

    return m12 + m3456


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("Usage: python3 cube3.py demo")
        print("  Runs a deterministic scramble-and-solve demo.")
