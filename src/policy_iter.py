from value_iter import recreate_path
from value_iter import init_pi, init_v
from grid import GridWorld, make_random_grid
from typing import List, Tuple, Iterable, Set
import numpy as np

#predefinitions to make takings actions easier
L=0
R=1
U=2
D=3
S=4

action_symbols = {0: "L", 1: "R", 2: "U", 3: "D", 4: "S"}

DISCOUNT = 0.9

def evaluate_policy(pi: np.ndarray, env: GridWorld, 
                    error: float = 1e-10) -> np.ndarray:
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            a      = int(pi[s])
            v_new  = env.reward[s, a] + DISCOUNT * V[env.next_state[s, a]]
            delta  = max(delta, abs(v_new - V[s]))
            V[s]   = v_new
        if delta < error:
            break
    return V


def policy_iteration(env: GridWorld):
    pi = np.zeros(env.nS, dtype=int)      # arbitrary start, say all 0 = Left
    while True:
        V = evaluate_policy(pi, env)
        policy_stable = True
        for s in range(env.nS):
            # one-step look-ahead for all actions
            q_sa = env.reward[s] + DISCOUNT * V[env.next_state[s]]
            best_a = int(np.argmax(q_sa))
            if best_a != pi[s]:
                policy_stable = False
                pi[s] = best_a
        if policy_stable:
            break
    return pi, V

if __name__ == "__main__":
    env = make_random_grid(10, 10, 30, seed=45)
    pi, V = policy_iteration(env)
    path = recreate_path(pi, env)
    print("Randomly generated environment:\n")
    env.render()
    print("Path:\n")
    print(path)


