from __future__ import annotations
import numpy as np
from tabulate import tabulate
from typing import List, Tuple, Iterable, Set

Action = int  # alias for readability (0‑4)
State  = int  # flattened index

# Displacement vectors for actions: L, R, U, D, Stay
_DRDC = [ (0, -1), (0, 1), (-1, 0), (1, 0), (0, 0) ]

HOLE_REWARD  = -1.0
GOAL_REWARD  = +1.0
STEP_REWARD  =  0.0

class GridWorld:
    """Deterministic tabular environment suitable for DP algorithms."""
    def __init__(self,
                 rows: int,
                 cols: int,
                 start: Tuple[int, int],
                 goal: Tuple[int, int],
                 holes: Iterable[Tuple[int, int]] = ()):        
        self.rows, self.cols = rows, cols
        self.nS, self.nA = rows * cols, 5

        # Helper lambdas ------------------------------------------------------
        self._to_id   = lambda rc: rc[0] * self.cols + rc[1]
        self._to_rc   = lambda s: divmod(s, self.cols)

        # Convert & store special cells --------------------------------------
        self.start: State = self._to_id(start)
        self.goal:  State = self._to_id(goal)
        self.holes: Set[State] = {self._to_id(h) for h in holes if h != goal and h != start}

        # Tables --------------------------------------------------------------
        self.next_state = np.zeros((self.nS, self.nA), dtype=np.int32)
        self.reward     = np.zeros((self.nS, self.nA), dtype=np.float32)
        self._build_transition_tables()

    # ---------------------------------------------------------------------
    def _build_transition_tables(self):
        for s in range(self.nS):
            r, c = self._to_rc(s)
            for a, (dr, dc) in enumerate(_DRDC):
                nr, nc = r + dr, c + dc
                # Off‑grid? => stay, reward -1
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    self.next_state[s, a] = s
                    self.reward[s, a]     = HOLE_REWARD  # same as off‑grid penalty
                    continue

                s_next = self._to_id((nr, nc))
                self.next_state[s, a] = s_next

                if s_next == self.goal:
                    self.reward[s, a] = GOAL_REWARD
                elif s_next in self.holes:
                    self.reward[s, a] = HOLE_REWARD
                else:
                    self.reward[s, a] = STEP_REWARD

    # ------------------------------------------------------------------ env API
    def reset(self) -> State:
        return self.start

    def step(self, s: State, a: Action) -> Tuple[State, float]:
        """Return (next_state, reward).  No done flag because episodes never end."""
        return int(self.next_state[s, a]), float(self.reward[s, a])

    # -------------------------------------------------------------- visualisation
    def render(self, agent_state: State | None = None):
        symbols = []
        for s in range(self.nS):
            if s == agent_state:
                symbols.append("A")
            elif s == self.start:
                symbols.append("S")
            elif s == self.goal:
                symbols.append("G")
            elif s in self.holes:
                symbols.append("H")
            else:
                symbols.append("·")
        # reshape & draw ------------------------------------------------------
        rows = [ symbols[i:i+self.cols] for i in range(0, self.nS, self.cols) ]
        print(tabulate(rows, tablefmt="grid"))

# ---------------------------------------------------------------------------
# Utility to build a random instance ----------------------------------------
# ---------------------------------------------------------------------------

def make_random_grid(rows: int,
                     cols: int,
                     n_holes: int = 0,
                     seed: int | None = None) -> GridWorld:
    rng = np.random.default_rng(seed)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    start = tuple(map(int, rng.choice(all_cells)))
    remaining = [cell for cell in all_cells if cell != start]
    goal  = tuple(map(int, rng.choice(remaining)))
    remaining.remove(goal)
    holes = [tuple(map(int, h)) for h in rng.choice(remaining, size=min(n_holes, len(remaining)), replace=False)]
    return GridWorld(rows, cols, start, goal, holes)


if __name__ == "__main__":
    env = make_random_grid(4, 4, n_holes=3, seed=40)
    print("Randomly generated environment:\n")
    env.render()
    
