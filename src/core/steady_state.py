import numpy as np
from typing import Optional

def steady_state_probabilities(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Solve for steady-state probabilities using transition matrix A.
    
    Args:
        A (numpy.ndarray): Transition rate matrix
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        
    Returns:
        numpy.ndarray: Steady-state probabilities
    """
    num_states = A.shape[0]
    
    # Initialize probability vector
    pi = np.ones(num_states) / num_states
    
    # Iterative solution
    for _ in range(max_iter):
        pi_new = np.zeros(num_states)
        
        # Update probabilities
        for j in range(num_states):
            pi_new[j] = -sum(pi[i] * A[i,j] for i in range(num_states)) / A[j,j]
        
        # Normalize
        pi_new = pi_new / np.sum(pi_new)
        
        # Check convergence
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        
        pi = pi_new
        
    raise RuntimeError(f"Failed to converge after {max_iter} iterations")

def compute_mm_n_probabilities(N: int, lambda_rate: float, mu_rate: float) -> np.ndarray:
    """Compute M/M/N queue probabilities for initialization.
    
    Args:
        N (int): Number of servers
        lambda_rate (float): Arrival rate
        mu_rate (float): Service rate
        
    Returns:
        numpy.ndarray: M/M/N queue probabilities
    """
    rho = lambda_rate / (N * mu_rate)
    
    # Compute p0
    sum_term = sum((N * rho)**n / np.math.factorial(n) for n in range(N))
    p0 = 1 / (sum_term + (N * rho)**N / (np.math.factorial(N) * (1 - rho)))
    
    # Compute other probabilities
    p = np.zeros(N + 1)
    p[0] = p0
    
    for n in range(1, N + 1):
        p[n] = (N * rho)**n * p0 / np.math.factorial(n)
    
    return p

def initialize_probabilities(N: int, method: str = 'uniform') -> np.ndarray:
    """Initialize probability estimates for iteration.
    
    Args:
        N (int): Number of units
        method (str): Initialization method ('uniform' or 'weighted')
        
    Returns:
        numpy.ndarray: Initial probability estimates
    """
    num_states = 2**N
    
    if method == 'uniform':
        return np.ones(num_states) / num_states
    
    elif method == 'weighted':
        # Weight by number of busy units
        weights = np.array([bin(i).count('1') for i in range(num_states)])
        weights = weights + 1  # Avoid zero weights
        return weights / np.sum(weights)
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")