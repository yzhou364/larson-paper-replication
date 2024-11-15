import numpy as np
from typing import List, Tuple

def initialize_transition_matrix(N: int, lambda_rate: float, mu_rate: float) -> np.ndarray:
    """Initialize the transition matrix for N units.
    
    Args:
        N (int): Number of units
        lambda_rate (float): Arrival rate
        mu_rate (float): Service rate
        
    Returns:
        numpy.ndarray: Transition rate matrix
    """
    num_states = 2**N
    A = np.zeros((num_states, num_states))
    
    # Generate all possible states
    for i in range(num_states):
        state = format(i, f'0{N}b')  # Binary representation
        num_busy = state.count('1')
        
        # Upward transitions (unit becomes busy)
        if num_busy < N:
            rate = lambda_rate / (N - num_busy)
            for n in range(N):
                if state[n] == '0':
                    new_state = list(state)
                    new_state[n] = '1'
                    j = int(''.join(new_state), 2)
                    A[i, j] = rate
        
        # Downward transitions (unit becomes available)
        if num_busy > 0:
            for n in range(N):
                if state[n] == '1':
                    new_state = list(state)
                    new_state[n] = '0'
                    j = int(''.join(new_state), 2)
                    A[i, j] = mu_rate
        
        # Diagonal elements
        A[i, i] = -np.sum(A[i, :])
    
    return A

def compress_matrix(A: np.ndarray) -> Tuple[dict, np.ndarray]:
    """Compress transition matrix using efficient storage scheme.
    
    Args:
        A (numpy.ndarray): Full transition matrix
        
    Returns:
        Tuple[dict, numpy.ndarray]: Compressed data and mapping array
    """
    N = int(np.log2(A.shape[0]))
    compressed_data = {}
    map_array = np.zeros(2**N, dtype=int)
    
    current_index = 0
    for i in range(2**N):
        map_array[i] = current_index
        state_weight = format(i, f'0{N}b').count('1')
        
        # Store non-zero elements
        for j in range(2**N):
            if A[i,j] != 0:
                compressed_data[current_index + state_weight] = (j, A[i,j])
        
        current_index += state_weight + 1
    
    return compressed_data, map_array

def decompress_matrix(compressed_data: dict, map_array: np.ndarray, N: int) -> np.ndarray:
    """Decompress matrix from efficient storage format.
    
    Args:
        compressed_data (dict): Compressed transition data
        map_array (numpy.ndarray): Mapping array
        N (int): Number of units
        
    Returns:
        numpy.ndarray: Full transition matrix
    """
    size = 2**N
    A = np.zeros((size, size))
    
    for i in range(size):
        map_index = map_array[i]
        state_weight = format(i, f'0{N}b').count('1')
        
        for k in range(state_weight + 1):
            if map_index + k in compressed_data:
                j, value = compressed_data[map_index + k]
                A[i,j] = value
    
    return A