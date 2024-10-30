import numpy as np
from hypercube import generate_states

def initialize_transition_matrix(N, lambda_rate, mu_rate):
    """
    Initialize the transition matrix for N units.
    """
    num_states = 2**N
    A = np.zeros((num_states, num_states))
    states = generate_states(N)
    
    for i, state in enumerate(states):
        busy_units = sum(state)
        for j in range(N):
            new_state = list(state)
            if state[j] == 0:
                new_state[j] = 1
                new_state_index = states.index(tuple(new_state))
                A[i, new_state_index] = lambda_rate / (N - busy_units)
            elif state[j] == 1:
                new_state[j] = 0
                new_state_index = states.index(tuple(new_state))
                A[i, new_state_index] = mu_rate  
    return A
