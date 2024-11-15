from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional

class BaseHypercubeModel(ABC):
    """Abstract base class for hypercube queuing models."""
    
    def __init__(self, N: int, J: int, lambda_rate: float, mu_rate: float = 1.0):
        """
        Initialize base hypercube model.
        
        Args:
            N (int): Number of units
            J (int): Number of geographical atoms
            lambda_rate (float): Call arrival rate
            mu_rate (float): Service rate
        """
        self.N = N
        self.J = J
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.rho = lambda_rate / (N * mu_rate)
        
        # Initialize matrices
        self.T = np.zeros((J, J))  # Travel time matrix
        self.L = np.zeros((N, J))  # Location probability matrix
        self.f = np.ones(J) / J    # Demand distribution
        
        # State space variables
        self.transition_matrix = None
        self.steady_state_probs = None
        
    @abstractmethod
    def setup_linear_command(self, district_length: float = 1.0):
        """Setup linear command configuration."""
        pass
    
    @abstractmethod
    def compute_performance_measures(self) -> Dict:
        """Compute system performance measures."""
        pass
    
    @abstractmethod
    def get_optimal_dispatch(self, state: List[int], atom: int) -> int:
        """Get optimal unit to dispatch for given state and atom."""
        pass
    
    def validate_inputs(self):
        """Validate model inputs."""
        if self.N <= 0:
            raise ValueError("Number of units must be positive")
        if self.J <= 0:
            raise ValueError("Number of atoms must be positive")
        if self.lambda_rate <= 0:
            raise ValueError("Arrival rate must be positive")
        if self.mu_rate <= 0:
            raise ValueError("Service rate must be positive")
        if self.rho >= 1:
            raise ValueError("System utilization must be less than 1")
            
    def compute_travel_times(self):
        """Compute travel time matrix."""
        for i in range(self.J):
            for j in range(self.J):
                if i == j:
                    self.T[i,j] = 0.5  # Intra-atom travel time
                else:
                    self.T[i,j] = abs(i - j)  # Manhattan distance
                    
    def set_location_probabilities(self, district_assignments: List[List[int]]):
        """Set location probabilities based on district assignments."""
        for n in range(self.N):
            district = district_assignments[n]
            if district:  # If district is not empty
                self.L[n, district] = 1.0 / len(district)
                
    def update_demand_distribution(self, demand: np.ndarray):
        """Update demand distribution."""
        if demand.sum() > 0:
            self.f = demand / demand.sum()
        else:
            self.f = np.ones(self.J) / self.J
            
    def get_state_representation(self, state: int) -> List[int]:
        """Convert state number to binary representation."""
        return [int(x) for x in format(state, f'0{self.N}b')]
    
    def get_state_number(self, state: List[int]) -> int:
        """Convert binary state representation to state number."""
        return int(''.join(map(str, state)), 2)
    
    @abstractmethod
    def run(self) -> Dict:
        """Run model and return results."""
        pass