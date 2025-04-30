# visualisations.py  (full working version)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import colors
from grid import GridWorld                     # ← your env module

LOGFILE   = "policy_iter_log.npz"
MAX_STEPS = 15

# ------------ load snapshot data ---------------------------------
data     = np.load(LOGFILE)
pol_log  = data["policy"]          # (k, nS)
val_log  = data["value"]           # (k, nS)
rows, cols = int(data["rows"]), int(data["cols"])
nS         = rows * cols
start      = int(data["start"])
goal       = int(data["goal"])
holes      = set(data["holes"])

# -------- recreate the deterministic environment -----------------
def id2rc(s): return divmod(int(s), cols)
env = GridWorld(rows, cols,
                start=id2rc(start),
                goal =id2rc(goal),
                holes=[id2rc(h) for h in holes])

# -------- static background --------------------------------------
grid_img = np.zeros((rows, cols), int)
for h in holes: grid_img[id2rc(h)] = 1      # holes black
grid_img[id2rc(goal)]  = 2                  # goal  green
grid_img[id2rc(start)] = 3                  # start yellow
cmap  = colors.ListedColormap(["white","black","lightgreen","yellow"])
norm  = colors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap.N)

# -------- figure / axes ------------------------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(grid_img, cmap=cmap, norm=norm)
ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=0.5)
ax.set_xticklabels([]); ax.set_yticklabels([])

# write numeric state values
texts = [[ax.text(c, r, "", ha="center", va="center", fontsize=8)
          for c in range(cols)] for r in range(rows)]

agent_dot, = ax.plot([], [], "ro", markersize=8)

# -------- helper to roll out one policy --------------------------
# -------- helper to roll out one policy --------------------------
def rollout(policy):
    """
    Return the sequence of states when we follow `policy`
    (max MAX_STEPS), **skipping any “stay” actions** so the
    animation doesn’t waste time on them.
    """
    path = [start]
    s    = start
    for _ in range(MAX_STEPS):
        a = int(policy[s])

        # If the policy says “Stay”, stop this rollout early
        # and move on to the next stored policy.
        if a == 4:                # 4 = Stay
            break

        s = int(env.next_state[s, a])
        path.append(s)
        if s == goal:
            break
    return path


paths = [rollout(pi) for pi in pol_log]
offsets = np.cumsum([0] + [len(p)+1 for p in paths])  # +1 pause
total_frames = offsets[-1]

def update_values(v):
    vmat = v.reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            texts[r][c].set_text(f"{vmat[r,c]:.2f}")

# -------- animation ------------------------------------------------
def animate(frame):
    idx = np.searchsorted(offsets, frame, side="right") - 1
    step = frame - offsets[idx]
    policy = pol_log[idx]
    values = val_log[idx]
    if step == 0:
        update_values(values)
    path = paths[idx]
    if step < len(path):
        r, c = id2rc(path[step])
        agent_dot.set_data([c],[r])
    else:
        agent_dot.set_data([],[])
    ax.set_title(f"Policy {idx+1}/{len(paths)} (iter {(idx+1)*3})")
    return agent_dot, *sum(texts, [])

ani = anim.FuncAnimation(fig, animate, frames=total_frames,
                         interval=600, blit=False)
plt.show()
