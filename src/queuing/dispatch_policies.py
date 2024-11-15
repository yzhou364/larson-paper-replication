import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DispatchDecision:
    """Container for dispatch decision results."""
    unit: int  # Selected unit
    travel_time: float  # Expected travel time
    is_interdistrict: bool  # Whether dispatch is interdistrict
    dispatch_weight: float  # Decision weight/score
    alternative_units: List[int] = None  # Other possible units

class DispatchPolicy(ABC):
    """Abstract base class for dispatch policies."""
    
    def __init__(self, N: int, J: int):
        """Initialize dispatch policy.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
        """
        self.N = N
        self.J = J
        
    @abstractmethod
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> DispatchDecision:
        """Get unit to dispatch given current state and call location.
        
        Args:
            state (List[int]): Current system state
            atom (int): Calling atom
            T (np.ndarray): Travel time matrix
            L (np.ndarray): Location probability matrix
            
        Returns:
            DispatchDecision: Dispatch decision details
        """
        pass

class ModifiedCenterMassPolicy(DispatchPolicy):
    """Modified Center of Mass (MCM) dispatch policy."""
    
    def __init__(self, N: int, J: int, districts: List[List[int]]):
        """Initialize MCM policy.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
            districts (List[List[int]]): District assignments
        """
        super().__init__(N, J)
        self.districts = districts
        
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> DispatchDecision:
        """Get unit using MCM policy."""
        available_units = [n for n in range(self.N) if state[n] == 0]
        
        if not available_units:
            # All units busy, return artificial unit for zero-line
            # or random busy unit for infinite-line
            return DispatchDecision(
                unit=self.N,
                travel_time=float('inf'),
                is_interdistrict=True,
                dispatch_weight=0.0
            )
        
        # Compute mean travel times for available units
        travel_times = []
        for n in available_units:
            time = sum(L[n,i] * T[i,atom] for i in range(self.J))
            is_interdistrict = atom not in self.districts[n]
            travel_times.append((n, time, is_interdistrict))
        
        # Sort by travel time
        travel_times.sort(key=lambda x: x[1])
        
        # Select unit with minimum travel time
        best_unit, best_time, is_interdistrict = travel_times[0]
        
        # Find alternative units with similar travel times
        tolerance = 0.1  # 10% tolerance
        alternative_units = [
            unit for unit, time, _ in travel_times[1:]
            if abs(time - best_time) <= best_time * tolerance
        ]
        
        return DispatchDecision(
            unit=best_unit,
            travel_time=best_time,
            is_interdistrict=is_interdistrict,
            dispatch_weight=1.0,
            alternative_units=alternative_units
        )

class DistrictPreferencePolicy(DispatchPolicy):
    """District-based preference dispatch policy."""
    
    def __init__(self, N: int, J: int, districts: List[List[int]], 
                 allow_interdistrict: bool = True):
        """Initialize district preference policy.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
            districts (List[List[int]]): District assignments
            allow_interdistrict (bool): Whether to allow interdistrict dispatch
        """
        super().__init__(N, J)
        self.districts = districts
        self.allow_interdistrict = allow_interdistrict
        
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray) -> DispatchDecision:
        """Get unit using district preferences."""
        # First try to find available unit from atom's district
        district_unit = None
        for n in range(self.N):
            if atom in self.districts[n]:
                if state[n] == 0:  # Unit is available
                    district_unit = n
                    break
        
        if district_unit is not None:
            # Found available unit in district
            travel_time = sum(L[district_unit,i] * T[i,atom] 
                            for i in range(self.J))
            return DispatchDecision(
                unit=district_unit,
                travel_time=travel_time,
                is_interdistrict=False,
                dispatch_weight=1.0
            )
        
        if not self.allow_interdistrict:
            # No district unit available and interdistrict not allowed
            return DispatchDecision(
                unit=self.N,
                travel_time=float('inf'),
                is_interdistrict=True,
                dispatch_weight=0.0
            )
        
        # Use MCM for out-of-district dispatch
        mcm = ModifiedCenterMassPolicy(self.N, self.J, self.districts)
        return mcm.get_dispatch_unit(state, atom, T, L)

class WorkloadBalancingPolicy(DispatchPolicy):
    """Workload-aware dispatch policy."""
    
    def __init__(self, N: int, J: int, districts: List[List[int]], 
                 workload_weight: float = 0.3):
        """Initialize workload balancing policy.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
            districts (List[List[int]]): District assignments
            workload_weight (float): Weight given to workload balance (0-1)
        """
        super().__init__(N, J)
        self.districts = districts
        self.workload_weight = workload_weight
        self.mcm = ModifiedCenterMassPolicy(N, J, districts)
    
    def get_dispatch_unit(self, state: List[int], atom: int,
                         T: np.ndarray, L: np.ndarray,
                         workloads: Optional[np.ndarray] = None) -> DispatchDecision:
        """Get unit considering both travel time and workload."""
        # Get MCM decision first
        mcm_decision = self.mcm.get_dispatch_unit(state, atom, T, L)
        
        if workloads is None or mcm_decision.unit == self.N:
            return mcm_decision
        
        available_units = [n for n in range(self.N) if state[n] == 0]
        if not available_units:
            return mcm_decision
        
        # Compute combined score for each available unit
        scores = []
        for n in available_units:
            # Travel time score (normalized)
            travel_time = sum(L[n,i] * T[i,atom] for i in range(self.J))
            time_score = mcm_decision.travel_time / travel_time
            
            # Workload score (inverse of current workload)
            workload_score = 1 - (workloads[n] / np.max(workloads))
            
            # Combined score
            combined_score = (
                (1 - self.workload_weight) * time_score +
                self.workload_weight * workload_score
            )
            
            scores.append((n, combined_score, travel_time))
        
        # Select unit with best combined score
        best_unit, best_score, travel_time = max(scores, key=lambda x: x[1])
        
        return DispatchDecision(
            unit=best_unit,
            travel_time=travel_time,
            is_interdistrict=atom not in self.districts[best_unit],
            dispatch_weight=best_score
        )

def create_dispatch_policy(policy_type: str, N: int, J: int, 
                         districts: List[List[int]], **kwargs) -> DispatchPolicy:
    """Factory function for creating dispatch policies.
    
    Args:
        policy_type (str): Type of policy to create
        N (int): Number of units
        J (int): Number of atoms
        districts (List[List[int]]): District assignments
        **kwargs: Additional policy-specific parameters
        
    Returns:
        DispatchPolicy: Configured dispatch policy
    """
    policies = {
        'mcm': ModifiedCenterMassPolicy,
        'district': DistrictPreferencePolicy,
        'workload': WorkloadBalancingPolicy
    }
    
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}")
        
    policy_class = policies[policy_type]
    
    if policy_type == 'workload':
        return policy_class(N, J, districts, 
                          workload_weight=kwargs.get('workload_weight', 0.3))
    elif policy_type == 'district':
        return policy_class(N, J, districts,
                          allow_interdistrict=kwargs.get('allow_interdistrict', True))
    else:
        return policy_class(N, J, districts)