import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

class DispatchPolicy(ABC):
    """Abstract base class for dispatch policies."""
    
    def __init__(self, N: int, J: int):
        self.N = N  # Number of units
        self.J = J  # Number of atoms
        
    @abstractmethod
    def get_dispatch_unit(self, state: List[int], atom: int, 
                         T: np.ndarray, L: np.ndarray) -> int:
        """Get unit to dispatch given current state and call location."""
        pass

class ModifiedCenterMassPolicy(DispatchPolicy):
    """Modified Center of Mass (MCM) dispatch policy."""
    
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> int:
        """Get unit using MCM policy."""
        available_units = [n for n in range(self.N) if state[n] == 0]
        
        if not available_units:
            return self.N  # Return artificial unit N+1
        
        # Compute mean travel times for available units
        travel_times = []
        for n in available_units:
            time = sum(L[n,i] * T[i,atom] for i in range(self.J))
            travel_times.append((n, time))
            
        # Return unit with minimum travel time
        return min(travel_times, key=lambda x: x[1])[0]

class FixedPreferencePolicy(DispatchPolicy):
    """Fixed preference order dispatch policy."""
    
    def __init__(self, N: int, J: int, preferences: List[List[int]]):
        """
        Initialize with preference orders.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
            preferences (List[List[int]]): Preferred unit order for each atom
        """
        super().__init__(N, J)
        self.preferences = preferences
        
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> int:
        """Get unit using fixed preferences."""
        for unit in self.preferences[atom]:
            if state[unit] == 0:  # Unit is available
                return unit
        return self.N  # Return artificial unit N+1

class DistrictDispatchPolicy(DispatchPolicy):
    """District-based dispatch policy."""
    
    def __init__(self, N: int, J: int, district_assignments: List[List[int]]):
        """
        Initialize with district assignments.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
            district_assignments (List[List[int]]): Atom assignments for each unit
        """
        super().__init__(N, J)
        self.district_assignments = district_assignments
        
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> int:
        """Get unit using district-based policy."""
        # First try district's primary unit
        for n in range(self.N):
            if atom in self.district_assignments[n] and state[n] == 0:
                return n
                
        # If primary unit unavailable, use MCM for backup
        return ModifiedCenterMassPolicy(self.N, self.J).get_dispatch_unit(
            state, atom, T, L
        )

class RandomizedPolicy(DispatchPolicy):
    """Randomized dispatch policy for comparison."""
    
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> int:
        """Get unit using randomized selection."""
        available_units = [n for n in range(self.N) if state[n] == 0]
        
        if not available_units:
            return self.N  # Return artificial unit N+1
            
        return np.random.choice(available_units)

def create_dispatch_policy(policy_type: str, N: int, J: int, **kwargs) -> DispatchPolicy:
    """Factory function for creating dispatch policies."""
    policies = {
        'mcm': ModifiedCenterMassPolicy,
        'fixed': FixedPreferencePolicy,
        'district': DistrictDispatchPolicy,
        'random': RandomizedPolicy
    }
    
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}")
        
    policy_class = policies[policy_type]
    
    if policy_type == 'fixed':
        return policy_class(N, J, kwargs['preferences'])
    elif policy_type == 'district':
        return policy_class(N, J, kwargs['district_assignments'])
    else:
        return policy_class(N, J)