import numpy as np


def simulate_RR_uniform_flavor(policy, T=3600):
    # policy: a list of choice probability, same policy for all restaurants
    # reward function: currently uniform across time and all restaurants
    policies = np.tile(policy, 4).reshape((4, 4), order='C')
    reward = 0
    probs = [0, 0.2, 0.8, 1]
    wait_time = [7, 5, 3, 1]
    reward_time = 2
    steps = 0
    accepts = []
    while T < 3600:
        tone = np.random.randint(4)
        restaurant = steps % 4
        # choice
        if np.random.random() < policies[restaurant][tone]:
            # accept, outcome
            T -= wait_time[tone]
            accepts.append(steps)
            if np.random.random() < probs[tone]:
                reward += 1
                T -= reward_time
        steps += 1
    return reward


def policy_optimize(T, k_reps=10):
    policy = [0.5] * 4
    objective = lambda plc: sum(simulate_RR_uniform_flavor(plc, T) for i in range(k_reps)) / k_reps
    # scipy optimize objective function with the argument policy
    optimal = None
    return optimal
