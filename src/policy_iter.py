from value_iter import recreate_path
from grid import GridWorld, make_random_grid
import numpy as np

# -------------------------------- constants -------------------------------
L, R, U, D, S = range(5)
DISCOUNT      = 0.99
LOGFILE       = "policy_iter_log.npz"          # <-- NEW
# --------------------------------------------------------------------------

def evaluate_policy(pi: np.ndarray, env: GridWorld,
                    error: float = 1e-10) -> np.ndarray:
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            a     = int(pi[s])
            v_new = env.reward[s, a] + DISCOUNT * V[env.next_state[s, a]]
            delta = max(delta, abs(v_new - V[s]))
            V[s]  = v_new
        if delta < error:
            break
    return V


def policy_iteration(env: GridWorld):
    pi = np.zeros(env.nS, dtype=int)      # start with all “Left”

    # --------------------------- NEW: log containers ----------------------
    policy_log = []          # list[(nS,) int]
    value_log  = []          # list[(nS,) float]
    env_meta   = {           # saved once for the visualiser
        "rows" : env.rows,
        "cols" : env.cols,
        "start": env.start,
        "goal" : env.goal,
        "holes": np.array(sorted(env.holes), dtype=int)
    }
    # ---------------------------------------------------------------------

    iterations = 0
    while True:
        V = evaluate_policy(pi, env)

        # store iterations
        if iterations % 2 == 0:
            policy_log.append(np.copy(pi))
            value_log .append(np.copy(V))


        policy_stable = True
        for s in range(env.nS):
            q_sa   = env.reward[s] + DISCOUNT * V[env.next_state[s]]
            best_a = int(np.argmax(q_sa))
            if best_a != pi[s]:
                policy_stable = False
                pi[s] = best_a
        if policy_stable:
            break
        iterations += 1

    print(f"Number of iterations: {iterations}")

    # ------------- NEW: dump the log to a compressed npz ----------------
    np.savez_compressed(
        LOGFILE,
        policy=np.stack(policy_log),
        value =np.stack(value_log ),
        **env_meta
    )
    # --------------------------------------------------------------------

    return pi, V


if __name__ == "__main__":
    env = make_random_grid(10, 10, n_holes=30, seed=20)
    pi_opt, V_opt = policy_iteration(env)
    print("Randomly generated environment:")
    env.render()
    print("Optimal policy (arrow grid):")
    print(recreate_path(pi_opt, env))
