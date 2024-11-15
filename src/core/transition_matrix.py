import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict

def initialize_transition_matrix(N: int, lambda_rate: float, mu_rate: float) -> np.ndarray:
    """Initialize the transition matrix for N units.
    
    Args:
        N (int): Number of units
        lambda_rate (float): Call arrival rate
        mu_rate (float): Service rate
        
    Returns:
        numpy.ndarray: Transition rate matrix
    """
    num_states = 2**N
    A = np.zeros((num_states, num_states))
    
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

def initialize_compressed_matrix(N: int) -> Tuple[Dict, np.ndarray]:
    """Initialize compressed storage for transition matrix.
    
    Args:
        N (int): Number of units
        
    Returns:
        Tuple[Dict, numpy.ndarray]: Compressed storage and mapping array
    """
    compressed_data = defaultdict(dict)
    map_array = np.zeros(2**N, dtype=int)
    
    current_index = 0
    for j in range(2**N):
        map_array[j] = current_index
        state_weight = format(j, f'0{N}b').count('1')
        current_index += state_weight + 1
        
    return compressed_data, map_array

def store_transition(compressed_data: Dict, map_array: np.ndarray, 
                    from_state: int, to_state: int, rate: float):
    """Store transition rate in compressed format.
    
    Args:
        compressed_data (Dict): Compressed storage dictionary
        map_array (numpy.ndarray): Mapping array
        from_state (int): Source state
        to_state (int): Destination state
        rate (float): Transition rate
    """
    if abs(rate) > 1e-10:  # Only store non-zero rates
        map_index = map_array[from_state]
        state_weight = format(from_state, f'0{len(map_array):b}b').count('1')
        compressed_data[map_index + state_weight][to_state] = rate

def get_transition_rate(compressed_data: Dict, map_array: np.ndarray, 
                       from_state: int, to_state: int) -> float:
    """Get transition rate from compressed storage.
    
    Args:
        compressed_data (Dict): Compressed storage dictionary
        map_array (numpy.ndarray): Mapping array
        from_state (int): Source state
        to_state (int): Destination state
        
    Returns:
        float: Transition rate
    """
    map_index = map_array[from_state]
    state_weight = format(from_state, f'0{len(map_array):b}b').count('1')
    return compressed_data[map_index + state_weight].get(to_state, 0.0)

def compute_upward_transitions(state: List[int], atom: int, N: int, lambda_rate: float,
                             dispatch_policy: callable) -> Dict[int, float]:
    """Compute upward transition rates for a given state and atom.
    
    Args:
        state (List[int]): Current system state
        atom (int): Calling atom
        N (int): Number of units
        lambda_rate (float): Call arrival rate
        dispatch_policy (callable): Function to determine optimal unit
        
    Returns:
        Dict[int, float]: Dictionary of {next_state: rate}
    """
    transitions = {}
    num_busy = sum(state)
    
    if num_busy < N:
        # Get optimal unit using dispatch policy
        unit = dispatch_policy(state, atom)
        
        if unit < N and state[unit] == 0:  # Valid unit and it's available
            new_state = state.copy()
            new_state[unit] = 1
            next_state = int(''.join(map(str, new_state)), 2)
            transitions[next_state] = lambda_rate / (N - num_busy)
            
    return transitions

def compute_downward_transitions(state: List[int], N: int, mu_rate: float) -> Dict[int, float]:
    """Compute downward transition rates for a given state.
    
    Args:
        state (List[int]): Current system state
        N (int): Number of units
        mu_rate (float): Service rate
        
    Returns:
        Dict[int, float]: Dictionary of {next_state: rate}
    """
    transitions = {}
    
    for n in range(N):
        if state[n] == 1:  # Unit is busy
            new_state = state.copy()
            new_state[n] = 0
            next_state = int(''.join(map(str, new_state)), 2)
            transitions[next_state] = mu_rate
            
    return transitions

def validate_transition_matrix(A: np.ndarray) -> bool:
    """Validate transition matrix properties.
    
    Args:
        A (numpy.ndarray): Transition matrix to validate
        
    Returns:
        bool: True if matrix is valid, False otherwise
    """
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        return False
    
    # Check if diagonal elements are negative
    if not all(A[i,i] < 0 for i in range(A.shape[0])):
        return False
    
    # Check if off-diagonal elements are non-negative
    if not all(A[i,j] >= 0 for i in range(A.shape[0]) 
               for j in range(A.shape[1]) if i != j):
        return False
    
    # Check if row sums are zero
    if not all(abs(sum(row)) < 1e-10 for row in A):
        return False
    
    return True

def get_matrix_density(A: np.ndarray) -> float:
    """Calculate matrix density (fraction of non-zero elements).
    
    Args:
        A (numpy.ndarray): Transition matrix
        
    Returns:
        float: Matrix density
    """
    nonzero = np.count_nonzero(A)
    total = A.size
    return nonzero / total