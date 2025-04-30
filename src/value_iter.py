from grid import GridWorld, make_random_grid
import numpy as np
from tabulate import tabulate
from typing import List, Tuple, Iterable, Set
import pathlib

CHECKPOINT = 5          # save every 20 iterations
LOGFILE    = "training_log.npz"

#predefinitions to make takings actions easier
L=0
R=1
U=2
D=3
S=4

action_symbols = {0: "L", 1: "R", 2: "U", 3: "D", 4: "S"}

DISCOUNT = 0.9
def init_v(env: GridWorld):
    return np.zeros(env.nS)

def init_pi(env: GridWorld):
    pi = np.empty(env.nS)
    for state in range(env.nS):
        pi[state] = S
    return pi

policy_log = []       # list[ np.ndarray ]  (nS,) int
value_log  = []       # list[ np.ndarray ]  (nS,) float
env_meta   = {}       # dict  (start/goal/holes/shape)  ← stored once


def recreate_path(pi: np.array, env: GridWorld):
    path_grid = np.zeros((env.rows, env.cols), dtype=object)
    ctr = 0
    
    for i in range(env.rows):
        for j in range(env.cols):
            # Convert numerical action to symbol
            action_idx = int(pi[ctr])
            path_grid[i][j] = action_symbols.get(action_idx, str(action_idx))
            ctr += 1
    
    # Format using tabulate for pretty printing
    table = tabulate(path_grid, tablefmt="grid")
    return table
        
        
def value_iteration(env: GridWorld):
    #initialize current policy: Stay everywhere    
    # pi is a nSx2 grid: for each state, what action we take. In this case, (0,0) for all
    #initialize v = 0 for all S
    # now make a q table with nSxnA
    #loop over s
        #for every a:
            #q[s,a] = immediate reward for the action from reward table + discount rate x v(next state for that action)
        #take maximum a and then set pi[s] = a
        #then set v[s] to the return of chosing this max a



    pi       = init_pi(env)
    v_table  = init_v(env)
    q_table  = np.zeros((env.nS, env.nA))
    diff     = float('inf')
    iters    = 0

    # LOG 0: environment descriptor (once)
    env_meta.update({
        "rows": env.rows,
        "cols": env.cols,
        "start": env.start,
        "goal": env.goal,
        "holes": np.array(sorted(env.holes), dtype=int)
    })

    while diff > 1e-4:
        v_new = np.copy(v_table)
        for s in range(env.nS):
            best = -np.inf
            for a in range(env.nA):
                q = env.reward[s, a] + 0.9 * v_table[env.next_state[s, a]]
                q_table[s, a] = q
                if q >= best:
                    best = q
                    pi[s] = a
            v_new[s] = best

        diff   = np.sum(np.abs(v_new - v_table))
        v_table = v_new
        iters  += 1

        # -------------------- LOG every 20 iterations -----------------
        if iters % CHECKPOINT == 0:
            policy_log.append(np.copy(pi))
            value_log .append(np.copy(v_table))

    print("Number of iterations:", iters)

    # -------------------- dump everything once VI is done ------------
    np.savez_compressed(
        LOGFILE,
        policy=np.stack(policy_log),
        value =np.stack(value_log ),
        **env_meta
    )

    return recreate_path(pi, env)


    # pi = init_pi(env)
    # v_table = init_v(env)
    # q_table = np.full((env.nS, env.nA), 0, dtype=float)
    # diff = float('inf')
    # iterations = 0
    # while diff>0.0001:
    #     v_table_new = np.copy(v_table)
    #     for state in range (env.nS):
    #         max_reward = float('-inf')
    #         for action in range (env.nA):
    #             qvalue = env.reward[state][action] + DISCOUNT*v_table[env.next_state[state][action]]
    #             q_table[state][action] = qvalue
    #             if qvalue>=max_reward:
    #                 max_reward = qvalue
    #                 pi[state] = action
    #         v_table_new[state] = max_reward
    #     diff = np.sum(np.absolute(v_table_new-v_table))
    #     v_table = np.copy(v_table_new)
    #     iterations += 1
    # ##Recreate path
    # path = recreate_path(pi, env)
    # print(f"Number of iterations: {iterations}")
    
    # return path


        

if __name__ == "__main__":
    env = make_random_grid(20, 20, n_holes=100, seed=42)
    print("Randomly generated environment:\n")
    env.render()
    path = value_iteration(env)
    print("Path:\n")
    print(path)


