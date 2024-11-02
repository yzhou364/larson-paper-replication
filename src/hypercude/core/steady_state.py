import numpy as np

def steady_state_probabilities(A):
    """
    Solve for steady-state probabilities using transition matrix A.
    """
    num_states = A.shape[0]
    A[-1, :] = 1
    b = np.zeros(num_states)
    b[-1] = 1
    pi = np.linalg.solve(A.T, b)
    return pi
