from synch import policy_iteration, random_policy, value_iteration
from asynch import in_place_value_iteration, prioritized_sweeping
from environment import GridWorldEnv
import time


ACTION_SYMBOLS = {
    "LEFT": "<",
    "RIGHT": ">",
    "UP": "^",
    "DOWN": "v",
}


def ask_int(prompt, default):
    try:
        text = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    if text == "":
        return default
    return int(text)


def render_policy(env, policy):
    for row in range(env.n_rows):
        cells = []
        for col in range(env.n_cols):
            state = env.pos_to_state(row, col)
            tile = env.get_tile(state)
            if tile in {"S", "G", "H"}:
                cells.append(tile)
            else:
                cells.append(ACTION_SYMBOLS[policy[state]])
        print(" ".join(cells))


def main():
    rows = ask_int("Rows", 4)
    cols = ask_int("Cols", 4)

    env = GridWorldEnv(rows=rows, cols=cols, hole_prob=0.2, seed=0)

    print("\nGenerated grid:")
    env.render()

    start = time.time()
    pi_vi, V_vi, V_count = value_iteration(env, 1e-5)
    print("\nValue iteration:", time.time() - start)
    print("Value iteration rounds: ", V_count)
    print("Estimated value from start:", V_vi[0])
    render_policy(env, pi_vi)

    start = time.time()
    pi_pi, V_pi, count = policy_iteration(env, random_policy(env))
    print("\nPolicy iteration:", time.time() - start)
    print("Policy improvement rounds:", count)
    print("Estimated value from start:", V_pi[0])
    render_policy(env, pi_pi)

    start = time.time()
    pi_ipvi, V_ipvi, ipv_count = in_place_value_iteration(env)
    print("\n In place value iteration:", time.time() - start)
    print("In place value iteration rounds: ", ipv_count)
    print("Estimated value from start:", V_ipvi[0])
    render_policy(env, pi_ipvi)

    start = time.time()
    sweep_pi, V_sweep, sweep_count = prioritized_sweeping(env)
    print("\nPrioritized sweeping:", time.time() - start)
    print("Prioritized sweeping updates: ", sweep_count)
    print("Estimated value from start:", V_sweep[0])
    render_policy(env, sweep_pi)
    

if __name__ == "__main__":
    main()
