import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

@dataclass
class StateTransition:
    """Represents a transition between system states."""
    from_state: int
    to_state: int
    rate: float
    type: str  # 'upward', 'downward', or 'diagonal'
    unit_changed: Optional[int] = None  # Unit that changed status

class StateMapper:
    """Maps and manages hypercube state representations."""
    
    def __init__(self, N: int):
        """Initialize state mapper.
        
        Args:
            N (int): Number of units/dimensions
        """
        self.N = N
        self.num_states = 2**N
        self.state_cache = {}  # Cache for binary representations
        self.adjacent_cache = {}  # Cache for adjacent states
        
    def get_binary_representation(self, state: int) -> List[int]:
        """Convert decimal state number to binary representation.
        
        Args:
            state (int): Decimal state number
            
        Returns:
            List[int]: Binary state representation
        """
        if state not in self.state_cache:
            self.state_cache[state] = [
                int(x) for x in format(state, f'0{self.N}b')
            ]
        return self.state_cache[state]
    
    def get_state_number(self, binary_state: List[int]) -> int:
        """Convert binary state representation to decimal number.
        
        Args:
            binary_state (List[int]): Binary state representation
            
        Returns:
            int: Decimal state number
        """
        return int(''.join(map(str, binary_state)), 2)
    
    def get_state_weight(self, state: int) -> int:
        """Get number of busy units (ones) in state.
        
        Args:
            state (int): State number
            
        Returns:
            int: Number of busy units
        """
        return bin(state).count('1')
    
    def get_adjacent_states(self, state: int) -> List[Tuple[int, str, int]]:
        """Get all states that are unit Hamming distance away.
        
        Args:
            state (int): Current state
            
        Returns:
            List[Tuple[int, str, int]]: List of (adjacent_state, transition_type, unit)
        """
        if state in self.adjacent_cache:
            return self.adjacent_cache[state]
        
        binary_state = self.get_binary_representation(state)
        adjacent_states = []
        
        # Check each unit
        for unit in range(self.N):
            new_state = binary_state.copy()
            if new_state[unit] == 0:
                # Unit becomes busy
                new_state[unit] = 1
                adjacent_states.append((
                    self.get_state_number(new_state),
                    'upward',
                    unit
                ))
            else:
                # Unit becomes available
                new_state[unit] = 0
                adjacent_states.append((
                    self.get_state_number(new_state),
                    'downward',
                    unit
                ))
                
        self.adjacent_cache[state] = adjacent_states
        return adjacent_states
    
    def get_states_by_weight(self, weight: int) -> List[int]:
        """Get all states with specified number of busy units.
        
        Args:
            weight (int): Number of busy units
            
        Returns:
            List[int]: States with specified weight
        """
        if not 0 <= weight <= self.N:
            raise ValueError(f"Weight must be between 0 and {self.N}")
            
        return [
            state for state in range(self.num_states)
            if self.get_state_weight(state) == weight
        ]
    
    def get_possible_transitions(self, state: int, 
                               lambda_rate: float, 
                               mu_rate: float) -> List[StateTransition]:
        """Get all possible transitions from given state.
        
        Args:
            state (int): Current state
            lambda_rate (float): Arrival rate
            mu_rate (float): Service rate
            
        Returns:
            List[StateTransition]: Possible transitions
        """
        transitions = []
        weight = self.get_state_weight(state)
        
        # Get adjacent states
        for adj_state, trans_type, unit in self.get_adjacent_states(state):
            rate = 0.0
            
            if trans_type == 'upward' and weight < self.N:
                # Arrival transition
                rate = lambda_rate / (self.N - weight)
            elif trans_type == 'downward':
                # Service completion
                rate = mu_rate
                
            if rate > 0:
                transitions.append(StateTransition(
                    from_state=state,
                    to_state=adj_state,
                    rate=rate,
                    type=trans_type,
                    unit_changed=unit
                ))
                
        # Add diagonal transition
        diagonal_rate = -(sum(t.rate for t in transitions))
        transitions.append(StateTransition(
            from_state=state,
            to_state=state,
            rate=diagonal_rate,
            type='diagonal'
        ))
        
        return transitions
    
    def get_state_sequence(self, start_state: int, 
                          transitions: List[int]) -> List[int]:
        """Generate sequence of states from transition sequence.
        
        Args:
            start_state (int): Initial state
            transitions (List[int]): List of units changing status
            
        Returns:
            List[int]: Sequence of states
        """
        sequence = [start_state]
        current_state = start_state
        
        for unit in transitions:
            binary_state = self.get_binary_representation(current_state)
            binary_state[unit] = 1 - binary_state[unit]  # Flip unit status
            current_state = self.get_state_number(binary_state)
            sequence.append(current_state)
            
        return sequence
    
    def is_valid_transition(self, from_state: int, to_state: int) -> bool:
        """Check if transition between states is valid (unit Hamming distance).
        
        Args:
            from_state (int): Source state
            to_state (int): Destination state
            
        Returns:
            bool: True if transition is valid
        """
        if from_state == to_state:
            return True  # Diagonal transition
            
        binary_from = self.get_binary_representation(from_state)
        binary_to = self.get_binary_representation(to_state)
        
        # Count differing bits
        differences = sum(b1 != b2 for b1, b2 in zip(binary_from, binary_to))
        return differences == 1
    
    def get_state_info(self, state: int) -> Dict:
        """Get detailed information about a state.
        
        Args:
            state (int): State number
            
        Returns:
            Dict: State information
        """
        binary_state = self.get_binary_representation(state)
        
        return {
            'decimal': state,
            'binary': binary_state,
            'weight': sum(binary_state),
            'busy_units': [i for i, busy in enumerate(binary_state) if busy],
            'available_units': [i for i, busy in enumerate(binary_state) if not busy]
        }
    
    def find_path(self, start_state: int, end_state: int) -> Optional[List[int]]:
        """Find shortest path between states using unit transitions.
        
        Args:
            start_state (int): Initial state
            end_state (int): Target state
            
        Returns:
            Optional[List[int]]: Sequence of units to change, None if no path
        """
        if start_state == end_state:
            return []
            
        visited = {start_state}
        queue = [(start_state, [])]
        
        while queue:
            current_state, path = queue.pop(0)
            
            # Try changing each unit
            for adj_state, _, unit in self.get_adjacent_states(current_state):
                if adj_state == end_state:
                    return path + [unit]
                    
                if adj_state not in visited:
                    visited.add(adj_state)
                    queue.append((adj_state, path + [unit]))
                    
        return None
    
    def validate_state(self, state: int) -> bool:
        """Validate state number.
        
        Args:
            state (int): State to validate
            
        Returns:
            bool: True if state is valid
        """
        return 0 <= state < self.num_states
    
    def clear_caches(self):
        """Clear state and adjacency caches."""
        self.state_cache.clear()
        self.adjacent_cache.clear()