# Synch-vs-Asynch-DP

The project compares several Dynamic Programming algorithms for solving a finite Markov Decision Process (MDP):

- Policy Iteration
- Synchronous Value Iteration
- In-place Value Iteration
- Prioritized Sweeping

The goal was to see how these different algorithms affect convergence speed and computational efficiency.

## Background

Dynamic Programming algorithms solve MDPs by repeatedly applying Bellman updates until the value function converges.

For a given state, the Bellman optimality backup is:

V(s) = max_a Σ P(s'|s,a) [R(s') + γV(s')]

All algorithms in this project ultimately compute the same optimal value function and policy. The primary difference is how and when updates are performed.

## Algorithms

### Policy Iteration

Policy Iteration alternates between two steps:
1. Policy evaluation
   - Compute the value function for the current policy
2. Policy Improvement
   - Improve the policy pi by choosing actions greedily with respect to the previously estimated value function
This process continues until the policy no longer changes.
Policy Iteration converges in relatively few iterations, but each iteration can be computationally expensive because it must do repeated full sweeps of the state space. It also must wait for the value function to converge before actually improving the policy.

Advantages:
- Simple and easy to understand.
- Guaranteed convergence under standard assumptions.
- Easy to implement.

Disadvantages:
- Newly computed values cannot influence other updates until the next sweep.
- Information propagates more slowly through the state space.

### Synchronous Value Iteration

Performs bellman optimality updates on every state during each sweep.

A temporary value function is used so that updates are based on values from the previous iteration.

Advantages:
- Simple and easy to understand.
- Guaranteed convergence under standard assumptions.
- Easy to implement.

Disadvantages:
- Newly computed values cannot influence other updates until the next sweep.
- Information propagates more slowly through the state space.

### In-Place (Asynchronous) Value Iteration

In-place Value Iteration updates states directly within the current value function.

As soon as a state's value is updated, future states in the same sweep can immediately benefit from that new information.

Advantages:
- Faster propagation of value information.
- Typically converges in fewer sweeps than synchronous value iteration.
- Requires no additional memory for a temporary value function.

### Prioritized Sweeping

Prioritized Sweeping focuses computation on states with large Bellman errors.

The algorithm:

1. Computes the Bellman error of each state.
2. Updates the state with the highest priority.
3. Reconsiders predecessor states because their values may now be inaccurate.
4. Repeats until convergence.

Rather than updating every state equally, prioritized sweeping directs computation toward the states most likely to produce meaningful changes.

Advantages:
- More efficient use of Bellman backups.
- Focuses computation where it matters most.
- Can significantly reduce the amount of work required for convergence.

---

## Results

All algorithms converged to the same optimal policy, but their convergence behavior differed.

### Policy Iteration

Policy Iteration required relatively few policy improvements before converging. However, each iteration was computationally expensive because the policy had to be evaluated repeatedly before improvement could occur.

### Synchronous Value Iteration

Synchronous Value Iteration converged reliably but required more sweeps because updates could not immediately benefit from newly computed values.

Information propagated one iteration at a time through the state space, making convergence slower than the asynchronous alternatives.

### In-Place Value Iteration

In-place Value Iteration was the strongest overall performer in my experiments.

Because updated values were immediately available, information propagated through the environment more quickly, reducing the number of sweeps required for convergence compared to synchronous value iteration.

### Prioritized Sweeping

Prioritized Sweeping demonstrated the benefits of intelligently selecting which states to update.

By focusing on states with large Bellman errors and propagating updates through predecessor states, it avoided many unnecessary backups and converged efficiently.

---
