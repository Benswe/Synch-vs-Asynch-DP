import random



def policy_evaluation(env, policy, theta=1e-5):
    # we need to return a dictionary giving the expected value of being in each state
    prev_V = {s: 0.0 for s in env.states}

    while True:
        V = {s: 0.0 for s in env.states}
        # loop through all states, if terminal skip
        for s in env.states:
            if env.is_terminal(s):
                continue
            for probability, next_state in env.get_transitions(s, policy[s]):
                reward = env.reward(next_state)
                gamma = env.gamma
                V[s] += probability * (reward + prev_V[next_state] * gamma)
        delta = max(abs(V[s] - prev_V[s]) for s in env.states)

        if delta < theta:
            break
        prev_V = V.copy()
    return V

def greedy_policy_improvement(env, V):
    # Should use the value function to calculate the best next action
    new_policy = {}
    # Goal: find best action for each state
    # we must compare values to find this
    for s in env.states:
        if env.is_terminal(s):
            continue
        best_action = None
        best_value = float("-INF")
        for action in env.actions:
            value = 0
            for probability, next_state in env.get_transitions(s, action):

                reward = env.reward(next_state)
                value += probability * (reward + env.gamma * V[next_state])
            if value > best_value:
                best_value = value
                best_action = action
        new_policy[s] = best_action
    return new_policy

def policy_iteration(env, policy):
    original_policy = policy
    count = 0
    while True:
        V = policy_evaluation(env, original_policy)

        new_policy = greedy_policy_improvement(env, V)
        count += 1
        if new_policy == original_policy:
            break
        original_policy = new_policy.copy()

    return new_policy, V, count
        





def value_iteration(env, theta):
    V = {s: 0.0 for s in env.states}
    policy = {}
    count = 0
    while True:
        V_old = V.copy()
        delta = 0
        for s in env.states:
            if env.is_terminal(s):
                continue
            best_action = None
            best_value = float("-INF")
            for a in env.actions:
                value = 0
                for probability, next_state in env.get_transitions(s, a):
                    reward = env.reward(next_state)
                    value += probability * (reward + env.gamma * V_old[next_state])
                if value > best_value:
                    best_value = value
                    best_action = a
            V[s] = best_value # max
            policy[s] = best_action # argmax
            delta = max(delta, abs(V_old[s] - V[s]))
        count += 1 
        if delta < theta:
            break
    return policy, V, count


def random_policy(env):
    """
    Build a random action for every non-terminal state.
    """
    return {
        s: random.choice(env.actions)
        for s in env.states
        if not env.is_terminal(s)
    }
