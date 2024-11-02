import numpy as np

def compute_workload(N, states, pi):
    """
    Compute the workload (fraction of time busy) for each unit.
    """
    workloads = np.zeros(N)
    for i, state in enumerate(states):
        for j in range(N):
            if state[j] == 1:
                workloads[j] += pi[i]
    return workloads

def compute_workload_imbalance(workloads):
    """
    Compute workload imbalance between the busiest and least busy units.
    """
    return np.max(workloads) - np.min(workloads)
