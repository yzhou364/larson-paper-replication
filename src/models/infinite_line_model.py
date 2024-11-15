import numpy as np
from typing import Dict, List, Optional
from .base_model import BaseHypercubeModel
from src.core.transition_matrix import initialize_transition_matrix
from src.core.steady_state import steady_state_probabilities
from src.core.performance import PerformanceMetrics

class InfiniteLineModel(BaseHypercubeModel):
    """Implementation of infinite-line capacity hypercube model."""
    
    def __init__(self, N: int, J: int, lambda_rate: float, mu_rate: float = 1.0):
        """
        Initialize infinite-line capacity model.
        
        Args:
            N (int): Number of units
            J (int): Number of geographical atoms
            lambda_rate (float): Call arrival rate
            mu_rate (float): Service rate
        """
        super().__init__(N, J, lambda_rate, mu_rate)
        self.queue_probs = None
        self.metrics = PerformanceMetrics()
        
    def setup_linear_command(self, district_length: float = 1.0):
        """Setup linear command with uniform atom sizes and districts.
        
        Args:
            district_length (float): Length of each district
        """
        self.validate_inputs()
        
        # Setup atoms and districts
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
        atom_length = district_length / atoms_per_district
        for i in range(self.J):
            for j in range(self.J):
                self.T[i,j] = abs(i - j) * atom_length
                
    def compute_performance_measures(self) -> Dict:
        """Compute performance measures for infinite-line capacity system."""
        if self.steady_state_probs is None:
            raise ValueError("Must run model before computing performance measures")
            
        measures = {}
        
        # Compute basic workloads
        workloads = self.metrics.compute_workload(
            self.N, 
            range(2**self.N), 
            self.steady_state_probs
        )
        
        # Compute queue metrics
        queue_metrics = self.metrics.compute_queue_metrics(
            self.N,
            self.lambda_rate,
            self.mu_rate,
            self.steady_state_probs
        )
        measures['queue_metrics'] = queue_metrics
        
        # Adjust workloads for queue
        p_queue = queue_metrics['p_all_busy']
        adjusted_workloads = workloads + (p_queue / self.N)
        measures['workloads'] = adjusted_workloads
        
        # Compute workload imbalance
        measures['workload_imbalance'] = self.metrics.compute_workload_imbalance(
            adjusted_workloads
        )
        
        # Compute dispatch probabilities
        dispatch_probs = self._compute_dispatch_probabilities()
        
        # Compute basic travel times
        base_travel_times = self.metrics.compute_travel_times(
            range(2**self.N),
            self.steady_state_probs,
            self.T,
            dispatch_probs
        )
        
        # Adjust travel times for queue
        queue_travel_time = self._compute_queue_travel_time()
        measures['travel_times'] = self._adjust_travel_times(
            base_travel_times,
            queue_travel_time,
            p_queue
        )
        
        # Compute interdistrict metrics
        measures['interdistrict'] = self.metrics.compute_interdistrict_fraction(
            range(2**self.N),
            self.steady_state_probs,
            self.district_assignments,
            dispatch_probs
        )
        
        # Compute regional metrics
        measures['regional'] = self.metrics.compute_regional_metrics(
            self.steady_state_probs,
            self.T,
            self.f,
            dispatch_probs
        )
        
        return measures
    
    def _compute_queue_travel_time(self) -> float:
        """Compute mean travel time for queued calls."""
        queue_time = 0
        for i in range(self.J):
            for j in range(self.J):
                queue_time += self.f[i] * self.f[j] * self.T[i,j]
        return queue_time
    
    def _adjust_travel_times(self, base_times: Dict[str, float],
                           queue_time: float, p_queue: float) -> Dict[str, float]:
        """Adjust travel times accounting for queued calls."""
        adjusted_times = {}
        for metric, value in base_times.items():
            if isinstance(value, (int, float)):
                adjusted_times[metric] = (1 - p_queue) * value + p_queue * queue_time
            elif isinstance(value, np.ndarray):
                adjusted_times[metric] = (1 - p_queue) * value + p_queue * queue_time
        return adjusted_times
    
    def get_optimal_dispatch(self, state: List[int], atom: int) -> int:
        """Get optimal unit to dispatch for given state and atom.
        
        Args:
            state (List[int]): Current system state
            atom (int): Atom requesting service
            
        Returns:
            int: Index of optimal unit to dispatch
        """
        available_units = [n for n in range(self.N) if state[n] == 0]
        
        if not available_units:
            # For infinite-line model, assign to random busy unit
            return np.random.randint(0, self.N)
        
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
                dispatch_probs[unit, atom] += prob * self.f[atom]
        
        return dispatch_probs
    
    def run(self) -> Dict:
        """Run infinite-line capacity model and return results."""
        # Initialize transition matrix
        self.transition_matrix = initialize_transition_matrix(
            self.N,
            self.lambda_rate,
            self.mu_rate
        )
        
        # Compute steady state probabilities
        self.steady_state_probs = steady_state_probabilities(self.transition_matrix)
        
        # Compute M/M/N queue probabilities for the infinite tail
        self._compute_queue_probabilities()
        
        # Compute and return performance measures
        return self.compute_performance_measures()
    
    def _compute_queue_probabilities(self):
        """Compute queue state probabilities for infinite queue."""
        rho = self.lambda_rate / (self.N * self.mu_rate)
        
        # Probability of all servers busy (from steady state)
        p_all_busy = sum(p for i, p in enumerate(self.steady_state_probs) 
                        if bin(i).count('1') == self.N)
        
        # Initialize queue probabilities array
        self.queue_probs = []
        
        # Compute probability for each queue length
        for k in range(50):  # Truncate at reasonable length
            p_k = p_all_busy * (rho ** k)
            self.queue_probs.append(p_k)
            
            # Break if probabilities become negligible
            if p_k < 1e-10:
                break
                
    def get_queue_length_distribution(self) -> Dict[int, float]:
        """Get probability distribution of queue lengths.
        
        Returns:
            Dict[int, float]: Mapping of queue lengths to probabilities
        """
        if self.queue_probs is None:
            raise ValueError("Must run model before getting queue distribution")
            
        return {k: p for k, p in enumerate(self.queue_probs)}
    
    def get_expected_queue_length(self) -> float:
        """Calculate expected queue length.
        
        Returns:
            float: Expected number of calls in queue
        """
        if self.queue_probs is None:
            raise ValueError("Must run model before calculating queue length")
            
        return sum(k * p for k, p in enumerate(self.queue_probs))
    
    def get_expected_system_size(self) -> float:
        """Calculate expected total number of calls in system.
        
        Returns:
            float: Expected number of calls in system (queued + in service)
        """
        # Number in service
        n_service = sum(bin(i).count('1') * p 
                       for i, p in enumerate(self.steady_state_probs))
        
        # Add expected queue length
        return n_service + self.get_expected_queue_length()
    
    def get_expected_wait_time(self) -> float:
        """Calculate expected wait time in queue.
        
        Returns:
            float: Expected waiting time for queued calls
        """
        # Using Little's Law: L = λW
        return self.get_expected_queue_length() / self.lambda_rate
    
    def get_expected_system_time(self) -> float:
        """Calculate expected total time in system.
        
        Returns:
            float: Expected total time in system (wait + service)
        """
        return self.get_expected_system_size() / self.lambda_rate
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive summary of system performance.
        
        Returns:
            Dict: Summary of key performance metrics
        """
        return {
            'utilization': self.lambda_rate / (self.N * self.mu_rate),
            'p_all_busy': sum(p for i, p in enumerate(self.steady_state_probs) 
                            if bin(i).count('1') == self.N),
            'mean_queue_length': self.get_expected_queue_length(),
            'mean_system_size': self.get_expected_system_size(),
            'mean_wait_time': self.get_expected_wait_time(),
            'mean_system_time': self.get_expected_system_time(),
            'queue_distribution': self.get_queue_length_distribution()
        }
    
    def validate_results(self) -> bool:
        """Validate model results for consistency.
        
        Returns:
            bool: True if results are consistent, False otherwise
        """
        # Check probability sum
        prob_sum = sum(self.steady_state_probs) + sum(self.queue_probs)
        if abs(prob_sum - 1.0) > 1e-6:
            return False
            
        # Check utilization
        rho = self.lambda_rate / (self.N * self.mu_rate)
        if rho >= 1:
            return False
            
        # Check queue length consistency
        if self.get_expected_queue_length() < 0:
            return False
            
        return True

    def analyze_sensitivity(self, parameter: str, 
                          values: List[float]) -> Dict[str, List[float]]:
        """Analyze system sensitivity to parameter changes.
        
        Args:
            parameter (str): Parameter to vary ('lambda', 'mu', or 'N')
            values (List[float]): Values to test
            
        Returns:
            Dict[str, List[float]]: Performance metrics for each value
        """
        original_lambda = self.lambda_rate
        original_mu = self.mu_rate
        original_N = self.N
        results = {
            'queue_length': [],
            'wait_time': [],
            'system_time': [],
            'utilization': []
        }
        
        try:
            for value in values:
                if parameter == 'lambda':
                    self.lambda_rate = value
                elif parameter == 'mu':
                    self.mu_rate = value
                elif parameter == 'N':
                    self.N = int(value)
                else:
                    raise ValueError(f"Invalid parameter: {parameter}")
                
                self.run()
                summary = self.get_performance_summary()
                
                results['queue_length'].append(summary['mean_queue_length'])
                results['wait_time'].append(summary['mean_wait_time'])
                results['system_time'].append(summary['mean_system_time'])
                results['utilization'].append(summary['utilization'])
                
        finally:
            # Restore original values
            self.lambda_rate = original_lambda
            self.mu_rate = original_mu
            self.N = original_N
            
        return results