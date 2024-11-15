import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

class ValidationLevel(Enum):
    """Validation level enumeration."""
    BASIC = "basic"
    STRICT = "strict"
    MINIMAL = "minimal"

@dataclass
class ValidationConfig:
    """Configuration for validation checks."""
    level: ValidationLevel = ValidationLevel.BASIC
    tolerance: float = 1e-10
    max_iterations: int = 1000
    check_convergence: bool = True
    check_consistency: bool = True
    validate_inputs: bool = True

class ModelValidator:
    """Validates hypercube model components and results."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize model validator.
        
        Args:
            config (Optional[ValidationConfig]): Validation configuration
        """
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
    def validate_inputs(self, N: int, J: int, lambda_rate: float, 
                       mu_rate: float) -> bool:
        """Validate basic model inputs.
        
        Args:
            N (int): Number of units
            J (int): Number of atoms
            lambda_rate (float): Arrival rate
            mu_rate (float): Service rate
            
        Returns:
            bool: True if inputs are valid
        """
        try:
            # Basic parameter validation
            if N <= 0:
                raise ValueError("Number of units must be positive")
            if J <= 0:
                raise ValueError("Number of atoms must be positive")
            if lambda_rate <= 0:
                raise ValueError("Arrival rate must be positive")
            if mu_rate <= 0:
                raise ValueError("Service rate must be positive")
                
            # System stability check
            rho = lambda_rate / (N * mu_rate)
            if rho >= 1 and self.config.level != ValidationLevel.MINIMAL:
                self.logger.warning(f"System may be unstable: ρ = {rho:.2f}")
                
            return True
            
        except ValueError as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            if self.config.level == ValidationLevel.STRICT:
                raise
            return False
        
    def validate_transition_matrix(self, A: np.ndarray) -> bool:
        """Validate transition rate matrix.
        
        Args:
            A (numpy.ndarray): Transition matrix
            
        Returns:
            bool: True if matrix is valid
        """
        try:
            # Check matrix shape
            if A.shape[0] != A.shape[1]:
                raise ValueError("Transition matrix must be square")
                
            # Check diagonal elements
            if not np.all(np.diag(A) <= 0):
                raise ValueError("Diagonal elements must be non-positive")
                
            # Check off-diagonal elements
            mask = ~np.eye(A.shape[0], dtype=bool)
            if not np.all(A[mask] >= 0):
                raise ValueError("Off-diagonal elements must be non-negative")
                
            # Check row sums
            row_sums = np.sum(A, axis=1)
            if not np.allclose(row_sums, 0, atol=self.config.tolerance):
                raise ValueError("Row sums must be zero")
                
            return True
            
        except ValueError as e:
            self.logger.error(f"Transition matrix validation failed: {str(e)}")
            if self.config.level == ValidationLevel.STRICT:
                raise
            return False
        
    def validate_steady_state(self, pi: np.ndarray, A: np.ndarray) -> bool:
        """Validate steady-state probabilities.
        
        Args:
            pi (numpy.ndarray): Steady state probabilities
            A (numpy.ndarray): Transition matrix
            
        Returns:
            bool: True if steady state is valid
        """
        try:
            # Check probability properties
            if not np.all(pi >= 0):
                raise ValueError("Probabilities must be non-negative")
                
            if not np.isclose(np.sum(pi), 1.0, atol=self.config.tolerance):
                raise ValueError("Probabilities must sum to 1")
                
            # Check balance equations
            balance = np.dot(pi, A)
            if not np.allclose(balance, 0, atol=self.config.tolerance):
                raise ValueError("Balance equations not satisfied")
                
            return True
            
        except ValueError as e:
            self.logger.error(f"Steady state validation failed: {str(e)}")
            if self.config.level == ValidationLevel.STRICT:
                raise
            return False
        
    def validate_district_configuration(self, districts: List[List[int]], 
                                     J: int) -> bool:
        """Validate district configuration.
        
        Args:
            districts (List[List[int]]): District assignments
            J (int): Number of atoms
            
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check atom coverage
            all_atoms = set()
            for district in districts:
                all_atoms.update(district)
                
            if len(all_atoms) != J:
                raise ValueError("Districts must cover all atoms")
                
            # Check for overlapping districts
            if self.config.level == ValidationLevel.STRICT:
                for i, dist1 in enumerate(districts):
                    for j, dist2 in enumerate(districts[i+1:], i+1):
                        if set(dist1) & set(dist2):
                            raise ValueError(f"Districts {i} and {j} overlap")
                            
            return True
            
        except ValueError as e:
            self.logger.error(f"District configuration validation failed: {str(e)}")
            if self.config.level == ValidationLevel.STRICT:
                raise
            return False
        
    def validate_performance_measures(self, measures: Dict) -> bool:
        """Validate computed performance measures.
        
        Args:
            measures (Dict): Performance measures
            
        Returns:
            bool: True if measures are valid
        """
        try:
            # Check workloads
            if 'workloads' in measures:
                workloads = measures['workloads']
                if not np.all(0 <= workloads) or np.any(workloads > 1):
                    raise ValueError("Workloads must be between 0 and 1")
                    
            # Check travel times
            if 'travel_times' in measures:
                if np.any(measures['travel_times'] < 0):
                    raise ValueError("Travel times must be non-negative")
                    
            # Check queue metrics
            if 'queue_metrics' in measures:
                queue_metrics = measures['queue_metrics']
                if queue_metrics.get('mean_queue_length', 0) < 0:
                    raise ValueError("Queue length must be non-negative")
                    
            return True
            
        except ValueError as e:
            self.logger.error(f"Performance measures validation failed: {str(e)}")
            if self.config.level == ValidationLevel.STRICT:
                raise
            return False
        
    def validate_convergence(self, values: List[float], 
                           threshold: Optional[float] = None) -> bool:
        """Check convergence of iterative calculations.
        
        Args:
            values (List[float]): Sequence of values
            threshold (Optional[float]): Convergence threshold
            
        Returns:
            bool: True if converged
        """
        if not self.config.check_convergence:
            return True
            
        threshold = threshold or self.config.tolerance
        
        if len(values) < 2:
            return False
            
        # Check relative change
        changes = np.abs(np.diff(values))
        return np.all(changes[-5:] < threshold)  # Last 5 iterations
        
    def validate_results(self, results: Dict) -> Dict[str, bool]:
        """Comprehensive validation of model results.
        
        Args:
            results (Dict): Model results
            
        Returns:
            Dict[str, bool]: Validation results by component
        """
        validation = {}
        
        # Validate steady state
        if 'steady_state_probs' in results and 'transition_matrix' in results:
            validation['steady_state'] = self.validate_steady_state(
                results['steady_state_probs'],
                results['transition_matrix']
            )
            
        # Validate performance measures
        if 'performance_measures' in results:
            validation['performance'] = self.validate_performance_measures(
                results['performance_measures']
            )
            
        # Validate district configuration
        if 'districts' in results and 'J' in results:
            validation['districts'] = self.validate_district_configuration(
                results['districts'],
                results['J']
            )
            
        # Check consistency
        if self.config.check_consistency:
            validation['consistency'] = self._check_result_consistency(results)
            
        return validation
    
    def _check_result_consistency(self, results: Dict) -> bool:
        """Check internal consistency of results.
        
        Args:
            results (Dict): Model results
            
        Returns:
            bool: True if results are consistent
        """
        try:
            # Check workload conservation
            if 'workloads' in results and 'lambda_rate' in results and 'mu_rate' in results:
                total_workload = np.sum(results['workloads'])
                expected_workload = results['lambda_rate'] / results['mu_rate']
                if not np.isclose(total_workload, expected_workload, 
                                rtol=0.1, atol=self.config.tolerance):
                    return False
                    
            # Check probability conservation
            if 'steady_state_probs' in results:
                pi = results['steady_state_probs']
                if not np.isclose(np.sum(pi), 1.0, atol=self.config.tolerance):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Consistency check failed: {str(e)}")
            return False