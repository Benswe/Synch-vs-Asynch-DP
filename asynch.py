import heapq

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
                    # bellman optimality equation
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
        







def bellman_backup(env, state, V):
    """
    Uses bellman optimal value function equation to find the value 
    and best action from state s
    
    """
    
    # use bellman optimality equation
    best_action = None
    best_value = float("-INF")
    for a in env.actions:
        val = 0
        for probability, next_state in env.get_transitions(state, a):
            reward = env.reward(next_state)
            val += probability * (reward + env.gamma*V[next_state])
        if val > best_value:
            best_value = val
            best_action = a

    return best_value, best_action

def bellman_error(env, state, V):
    """
    Helper for prioritized sweeping

    Finds bellman error for a given state
    """
    if env.is_terminal(state):
        return 0 # ignore if terminal
    
    value = V[state]
    bellman_value, _ = bellman_backup(env, state, V)
    return abs(bellman_value - value)

    
def highest_priority_state(queue):
    """
    Helper for prioritized sweeping

    Gives you the state with the highest error
    
    """
    if queue:
        negative_error, state = heapq.heappop(queue)
    return state
  

def prioritized_sweeping(env, theta=1e-5):
    """
    Why update every state equally when some states are much more wrong than others?
    """
    # predecessors are all states that could lead into a state s
    # these must also be updated if s is updated because the value function for 
    # the predecessor states is dependant on v[next_state], where next state is 
    # s
    predecessors = {s: set() for s in env.states}
    for s in env.states:
        if env.is_terminal(s):
            continue
        for a in env.actions:
            for probability, next_state in env.get_transitions(s, a):
                if probability > 0.0:
                    predecessors[next_state].add(s) 

    count = 0
    V = {s: 0.0 for s in env.states}
    policy = {}
    queue = []
    for s in env.states:
        if not env.is_terminal(s):
            error = bellman_error(env, s, V)
            heapq.heappush(queue, (-error, s))
    
    while len(queue) != 0:
        state = highest_priority_state(queue)
        error = bellman_error(env, state, V)
        if error < theta:
            continue

        V[state], policy[state] = bellman_backup(env, state, V)
        count += 1

        # now add predecessors to queue if necessary
        for p in predecessors[state]:

            error = bellman_error(env, p, V)
            # when close enought to convergence, no need to update
            # thus queue will go to 0
            if error > theta:
                heapq.heappush(queue, (-error, p))
    for s in env.states:
        if not env.is_terminal(s) and s not in policy:
            _, policy[s] = bellman_backup(env, s, V)
    return policy, V, count
        

            
