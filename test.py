from synch import policy_iteration, value_iteration
from environment import GridWorldEnv
import time

# 20x20 grid world
env = GridWorldEnv(grid = [
    "SFFFFFFFFFFFFFFFFFFF",
    "FFFHFFFFHFFFFFHFFFFF",
    "FHHFFFFFHFHFFFFFHFFF",
    "FFFFHFHFFFFFHHFFFFFF",
    "FHFFFFFHFFFFFHFFFFHF",
    "FFFFFHHFFFFHFFFFFHHF",
    "FFHFFFFFHFFFFFFHFFFF",
    "FFFFHFFFFFHHFFFFFHFF",
    "FHHFFFFHFFFFFFHFFFFF",
    "FFFFFHFFFFHHFFFFFHHF",
    "FFHFFFFFHFHFFFFFFHFF",
    "FFFFHHFFFFFFHFFFFFHF",
    "FHFFFFFHFFFFFHHFFFFF",
    "FFFFFHFHFFFFFFFHFFFF",
    "FFHHFFFFFHHFFFFFHFFF",
    "FFFFFHFFFFFFHFFFFHHF",
    "FHFHFFFFFFFHFFFFFHHF",
    "FFFFFHHFFFFHFFFFFFHF",
    "FFHFFFFFFFFFFFHFFFFF",
    "FFFFFFFFFFFFFFFFFFFG"
])
policy_table = {
    0: "RIGHT",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",

    4: "DOWN",
    6: "DOWN",

    8: "RIGHT",
    9: "RIGHT",
    10: "DOWN",

    13: "DOWN",
    14: "RIGHT"
}
start = time.time()
pi_vi, V_vi = value_iteration(env, 1e-5)
print("Value iteration:", time.time() - start)

start = time.time()
pi_pi, V_pi, count = policy_iteration(env, policy_table)
print("Policy iteration:", time.time() - start)