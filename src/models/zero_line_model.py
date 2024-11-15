import numpy as np
from typing import Dict, List, Optional
from .base_model import BaseHypercubeModel
from src.core.transition_matrix import initialize_transition_matrix
from src.core.steady_state import steady_state_probabilities
from src.core.performance import (
    compute_workload, 
    compute_workload_imbalance,
    compute_travel_times,
    compute_interdistrict_fraction
)

class ZeroLineModel(BaseHypercubeModel):
    """Implementation of zero-line capacity hypercube model."""
    
    def setup_linear_command(self, district_length: float = 1.0):
        """Setup linear command with uniform atom sizes and districts."""
        self.validate_inputs()
        
        # Setup atoms and districts
        atoms_per_district = self.J // self.N
        district_assignments = []
        
        for i in range(self.N):
            start_atom = i * atoms_per_district
            end_atom = start_atom + atoms_per_district
            district = list(range(start_atom, end_atom))
            district_assignments.append(district)
            
        # Set location probabilities
        self.set_location_probabilities(district_assignments)
        
        # Compute travel times
        atom_length = district_length / atoms_per_district
        for i in range(self.J):
            for j in range(self.J):
                self.T[i,j] = abs(i - j) * atom_length
                
    def compute_performance_measures(self) -> Dict:
        """Compute performance measures for zero-line capacity system."""
        if self.steady_state_probs is None:
            raise ValueError("Must run model before computing performance measures")
            
        measures = {}
        
        # Compute workloads
        workloads = compute_workload(self.N, 
                                   range(2**self.N), 
                                   self.steady_state_probs)
        measures['workloads'] = workloads
        
        # Compute workload imbalance
        measures['workload_imbalance'] = compute_workload_imbalance(workloads)
        
        # Compute travel times
        dispatch_probs = self._compute_dispatch_probabilities()
        measures['travel_times'] = compute_travel_times(
            range(2**self.N),
            self.steady_state_probs,
            self.T,
            dispatch_probs
        )
        
        # Compute interdistrict response metrics
        district_assignments = [[] for _ in range(self.N)]
        atoms_per_district = self.J // self.N
        for i in range(self.N):
            start_atom = i * atoms_per_district
            end_atom = start_atom + atoms_per_district
            district_assignments[i] = list(range(start_atom, end_atom))
            
        measures['interdistrict'] = compute_interdistrict_fraction(
            range(2**self.N),
            self.steady_state_probs,
            district_assignments,
            dispatch_probs
        )
        
        return measures
    
    def get_optimal_dispatch(self, state: List[int], atom: int) -> int:
        """Get optimal unit to dispatch for given state and atom."""
        available_units = [n for n in range(self.N) if state[n] == 0]
        
        if not available_units:
            return self.N  # Return artificial unit N+1
            
        # Compute mean travel times for available units
        travel_times = []
        for n in available_units:
            time = sum(self.L[n,i] * self.T[i,atom] for i in range(self.J))
            travel_times.append((n, time))
            
        # Return unit with minimum travel time
        return min(travel_times, key=lambda x: x[1])[0]
    
    def _compute_dispatch_probabilities(self) -> np.ndarray:
        """Compute probability matrix of unit-atom dispatch assignments."""
        dispatch_probs = np.zeros((self.N, self.J))
        
        for state_num in range(2**self.N):
            state = self.get_state_representation(state_num)
            prob = self.steady_state_probs[state_num]
            
            for atom in range(self.J):
                unit = self.get_optimal_dispatch(state, atom)
                if unit < self.N:  # Don't count artificial unit
                    dispatch_probs[unit, atom] += prob * self.f[atom]
                    
        return dispatch_probs
    
    def run(self) -> Dict:
        """Run zero-line capacity model and return results."""
        # Initialize transition matrix
        self.transition_matrix = initialize_transition_matrix(
            self.N, 
            self.lambda_rate, 
            self.mu_rate
        )
        
        # Compute steady state probabilities
        self.steady_state_probs = steady_state_probabilities(self.transition_matrix)
        
        # Compute and return performance measures
        return self.compute_performance_measures()