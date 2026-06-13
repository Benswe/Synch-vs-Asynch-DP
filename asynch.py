from environment import GridWorldEnv


env = GridWorldEnv()

def in_place_value_iteration(env, theta=1e-5):
    V = {s : 0.0 for s in env.states}
    policy = {}
    count = 0
    while True:
        # loop through each state
        delta = 0
        for s in env.states:
            old_V = V[s]
            # skip if terminal
            if env.is_terminal(s):
                continue
            best_action = None
            best_value = float("-INF")
            # for all actions available
            for a in env.actions:
                value = 0
                for probability, next_state in env.get_transitions(s, a): 
                    reward = env.reward(next_state)
                    value += probability * (reward + env.gamma*V[next_state])
                if value > best_value: 
                    best_value = value
                    best_action = a
            V[s] = best_value # max value  
            policy[s] = best_action
            delta = max(delta, abs(V[s] - old_V))
        count += 1
        if delta < theta:
            break
    return policy, V, count
        




def prioritized_sweeping(env, ):
    pass