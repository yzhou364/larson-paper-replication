import numpy as np
from typing import List, Optional

class BinarySequenceGenerator:
    """Generates unit-Hamming distance binary sequences for hypercube traversal."""
    
    def __init__(self, n_dimensions: int):
        """Initialize generator for n-dimensional hypercube.
        
        Args:
            n_dimensions (int): Number of dimensions/units
        """
        if n_dimensions <= 0:
            raise ValueError("Number of dimensions must be positive")
            
        self.n = n_dimensions
        self.sequence = None
        
    def generate(self) -> np.ndarray:
        """Generate complete binary sequence with unit-Hamming distance between adjacent elements."""
        sequence = [0, 1]  # Initial sequence for n=1
        
        for k in range(1, self.n):
            m_k = len(sequence)
            m_kplus1 = 2 * m_k
            
            # Generate new sequences by reflecting and adding 2^k
            new_elements = [(2**k + sequence[m_k-1-i]) for i in range(m_k)]
            sequence.extend(new_elements)
            
        self.sequence = np.array(sequence)
        return self.sequence
    
    def get_binary_representation(self, decimal_number: int) -> np.ndarray:
        """Convert decimal number to binary array representation."""
        if decimal_number < 0 or decimal_number >= 2**self.n:
            raise ValueError(f"Number must be between 0 and {2**self.n-1}")
            
        return np.array([int(x) for x in format(decimal_number, f'0{self.n}b')])
    
    def get_hamming_distance(self, state1: int, state2: int) -> int:
        """Calculate Hamming distance between two states."""
        bin1 = self.get_binary_representation(state1)
        bin2 = self.get_binary_representation(state2)
        return np.sum(bin1 != bin2)
    
    def get_adjacent_states(self, state: int) -> List[int]:
        """Get all states that are unit Hamming distance away from given state."""
        adjacent_states = []
        for i in range(self.n):
            # Flip each bit one at a time
            adjacent_state = state ^ (1 << i)
            adjacent_states.append(adjacent_state)
        return adjacent_states
    
    def verify_sequence(self) -> bool:
        """Verify that the generated sequence has unit Hamming distance between adjacent elements."""
        if self.sequence is None:
            return False
            
        for i in range(len(self.sequence) - 1):
            if self.get_hamming_distance(self.sequence[i], self.sequence[i + 1]) != 1:
                return False
        return True
    
    def get_weight(self, state: int) -> int:
        """Get the weight (number of ones) in the binary representation of a state."""
        return bin(state).count('1')
    
    def get_states_by_weight(self, weight: int) -> List[int]:
        """Get all states with a specific weight (number of ones)."""
        if weight < 0 or weight > self.n:
            raise ValueError(f"Weight must be between 0 and {self.n}")
            
        states = []
        for state in range(2**self.n):
            if self.get_weight(state) == weight:
                states.append(state)
        return states

    def get_changed_unit(self, state1: int, state2: int) -> Optional[int]:
        """Get the index of the unit that changed status between two adjacent states."""
        if self.get_hamming_distance(state1, state2) != 1:
            return None
            
        diff = state1 ^ state2  # XOR to find the changed bit
        return int(np.log2(diff))  # Position of the changed bit