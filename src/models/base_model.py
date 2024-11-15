from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.core.performance import PerformanceAnalyzer
from src.core.binary_sequences import BinarySequenceGenerator
from src.core.steady_state import SteadyStateCalculator
from src.core.transition_matrix import initialize_transition_matrix

@dataclass
class ModelConfig:
    """Configuration for hypercube model."""
    N: int  # Number of units
    J: int  # Number of atoms
    lambda_rate: float  # Arrival rate
    mu_rate: float = 1.0  # Service rate
    district_length: float = 1.0  # Length of each district
    dispatch_policy: str = 'mcm'  # Dispatch policy type

class BaseHypercubeModel(ABC):
    """Abstract base class for hypercube queuing models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize base hypercube model.
        
        Args:
            config (ModelConfig): Model configuration
        """
        self.config = config
        self.validate_config()
        
        # Initialize system parameters
        self.N = config.N
        self.J = config.J
        self.lambda_rate = config.lambda_rate
        self.mu_rate = config.mu_rate
        self.rho = self.lambda_rate / (self.N * self.mu_rate)
        
        # Initialize matrices
        self.T = np.zeros((self.J, self.J))  # Travel time matrix
        self.L = np.zeros((self.N, self.J))  # Location probability matrix
        self.f = np.ones(self.J) / self.J    # Demand distribution
        
        # Initialize analysis components
        self.binary_generator = BinarySequenceGenerator(self.N)
        self.performance_analyzer = PerformanceAnalyzer(
            self.N, self.J, self.lambda_rate, self.mu_rate
        )
        self.steady_state_calculator = SteadyStateCalculator(
            self.N, self.lambda_rate, self.mu_rate
        )
        
        # State space variables
        self.transition_matrix = None
        self.steady_state_probs = None
        self.district_assignments = None
        
    def validate_config(self):
        """Validate model configuration."""
        
        if self.config.N <= 0:
            raise ValueError("Number of units must be positive")
        if self.config.J <= 0:
            raise ValueError("Number of atoms must be positive")
        if self.config.lambda_rate <= 0:
            raise ValueError("Arrival rate must be positive")
        if self.config.mu_rate <= 0:
            raise ValueError("Service rate must be positive")
        if self.config.district_length <= 0:
            raise ValueError("District length must be positive")
            
    def setup_linear_command(self):
        """Setup linear command with uniform atom sizes and districts."""
        # Compute atoms per district
        atoms_per_district = self.J // self.N
        self.district_assignments = []
        
        # Create districts and assign atoms
        for i in range(self.N):
            start_atom = i * atoms_per_district
            end_atom = start_atom + atoms_per_district
            district = list(range(start_atom, end_atom))
            self.district_assignments.append(district)
            
            # Set uniform location probabilities within district
            self.L[i, start_atom:end_atom] = 1.0 / atoms_per_district
        
        # Compute travel time matrix
        atom_length = self.config.district_length / atoms_per_district
        for i in range(self.J):
            for j in range(self.J):
                if i == j:
                    self.T[i,j] = atom_length / 2  # Intra-atom travel time
                else:
                    self.T[i,j] = abs(i - j) * atom_length
    
    @abstractmethod
    def get_optimal_dispatch(self, state: List[int], atom: int) -> int:
        """Get optimal unit to dispatch for given state and atom."""
        pass
    
    @abstractmethod
    def compute_performance_measures(self) -> Dict:
        """Compute system performance measures."""
        pass
    
    def set_demand_distribution(self, demand: np.ndarray):
        """Set demand distribution across atoms.
        
        Args:
            demand (numpy.ndarray): Demand rates for each atom
        """
        if len(demand) != self.J:
            raise ValueError(f"Demand array must have length {self.J}")
        if not np.all(demand >= 0):
            raise ValueError("Demand rates must be non-negative")
            
        self.f = demand / np.sum(demand)
    
    def set_location_probabilities(self, location_probs: np.ndarray):
        """Set location probabilities for each unit.
        
        Args:
            location_probs (numpy.ndarray): Location probability matrix
        """
        if location_probs.shape != (self.N, self.J):
            raise ValueError(f"Location probability matrix must be {self.N}x{self.J}")
        if not np.allclose(np.sum(location_probs, axis=1), 1.0):
            raise ValueError("Location probabilities must sum to 1 for each unit")
            
        self.L = location_probs
    
    def get_state_representation(self, state: int) -> List[int]:
        """Convert state number to binary representation."""
        return [int(x) for x in format(state, f'0{self.N}b')]
    
    def get_state_number(self, state: List[int]) -> int:
        """Convert binary state representation to state number."""
        return int(''.join(map(str, state)), 2)
    
    def initialize_system(self):
        """Initialize system matrices and variables."""
        # Generate binary sequence
        self.binary_generator.generate()
        
        # Initialize transition matrix
        self.transition_matrix = initialize_transition_matrix(
            self.N, self.lambda_rate, self.mu_rate
        )
    
    def compute_steady_state(self, method: str = 'direct'):
        """Compute steady state probabilities.
        
        Args:
            method (str): Solution method ('direct', 'iterative', or 'mm_n')
        """
        if self.transition_matrix is None:
            raise ValueError("Must initialize system before computing steady state")
            
        self.steady_state_probs = self.steady_state_calculator.solve_steady_state(
            self.transition_matrix, method
        )
    
    @abstractmethod
    def run(self) -> Dict:
        """Run model and return results."""
        pass
    
    def get_performance_summary(self) -> Dict:
        """Get summary of system performance."""
        if self.steady_state_probs is None:
            raise ValueError("Must run model before getting performance summary")
            
        return {
            'utilization': self.rho,
            'workload_imbalance': self.performance_analyzer.compute_workload_imbalance(
                self.performance_analyzer.compute_workloads(
                    np.arange(2**self.N), self.steady_state_probs
                )
            ),
            'mean_travel_time': np.mean(self.T),
            'system_busy_prob': sum(
                p for i, p in enumerate(self.steady_state_probs)
                if bin(i).count('1') == self.N
            )
        }