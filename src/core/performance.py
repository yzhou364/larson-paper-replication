import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Container for system performance metrics."""
    workloads: np.ndarray
    travel_times: Dict[str, float]
    interdistrict_fractions: np.ndarray
    queue_metrics: Optional[Dict] = None

class PerformanceAnalyzer:
    """Analyze system performance for hypercube model."""
    
    def __init__(self, N: int, J: int, lambda_rate: float, mu_rate: float):
        """Initialize performance analyzer.
        
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
        
    def compute_workloads(self, states: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Compute workload (fraction of time busy) for each unit.
        
        Args:
            states (numpy.ndarray): Binary state representations
            pi (numpy.ndarray): Steady state probabilities
            
        Returns:
            numpy.ndarray: Workload for each unit
        """
        workloads = np.zeros(self.N)
        for i, state in enumerate(states):
            state_binary = format(state, f'0{self.N}b')
            for j in range(self.N):
                if state_binary[j] == '1':
                    workloads[j] += pi[i]
        return workloads
    
    def compute_workload_imbalance(self, workloads: np.ndarray) -> Dict[str, float]:
        """Compute workload imbalance metrics.
        
        Args:
            workloads (numpy.ndarray): Unit workloads
            
        Returns:
            Dict[str, float]: Imbalance metrics
        """
        return {
            'absolute': np.max(workloads) - np.min(workloads),
            'relative': (np.max(workloads) - np.min(workloads)) / np.mean(workloads),
            'std': np.std(workloads),
            'cv': np.std(workloads) / np.mean(workloads),
            'max_workload': np.max(workloads),
            'min_workload': np.min(workloads),
            'mean_workload': np.mean(workloads)
        }
    
    def compute_travel_times(self, states: np.ndarray, pi: np.ndarray, 
                           T: np.ndarray, dispatch_probs: np.ndarray) -> Dict[str, float]:
        """Compute various travel time metrics.
        
        Args:
            states (numpy.ndarray): System states
            pi (numpy.ndarray): Steady state probabilities
            T (numpy.ndarray): Travel time matrix (J x J)
            dispatch_probs (numpy.ndarray): Dispatch probabilities (N x J)
            
        Returns:
            Dict[str, float]: Travel time metrics
        """
        # Calculate expected travel time for each dispatch
        avg_travel_times = np.zeros_like(dispatch_probs)  # N x J matrix
        for n in range(self.N):  # For each unit
            for j in range(self.J):  # For each destination atom
                # Calculate weighted average travel time from all possible starting atoms
                avg_travel_times[n,j] = np.sum(T[:,j])  # Sum travel times to destination j

        # Compute overall average travel time
        avg_travel_time = np.sum(avg_travel_times * dispatch_probs)

        # Region-specific calculations
        atom_times = np.sum(dispatch_probs.T * avg_travel_times.T, axis=1)  # Average time to each atom
        unit_times = np.sum(dispatch_probs * avg_travel_times, axis=1)  # Average time for each unit

        return {
            'average': avg_travel_time,
            'by_unit': unit_times,
            'by_atom': atom_times,
            'max': np.max(T),
            'min': np.min(T[T > 0]),  # Minimum non-zero travel time
            'std': np.std(T[T > 0])
        }
    
    def compute_interdistrict_metrics(self, states: np.ndarray, pi: np.ndarray,
                                    districts: List[List[int]], 
                                    dispatch_matrix: np.ndarray) -> Dict[str, float]:
        """Compute interdistrict response metrics.
        
        Args:
            states (numpy.ndarray): System states
            pi (numpy.ndarray): Steady state probabilities
            districts (List[List[int]]): District definitions
            dispatch_matrix (numpy.ndarray): Dispatch probability matrix
            
        Returns:
            Dict[str, float]: Interdistrict metrics
        """
        interdistrict = np.zeros(self.N)
        total_dispatches = np.zeros(self.N)
        
        # Compute interdistrict dispatches for each unit
        for n in range(self.N):
            district_atoms = set(districts[n])
            for j in range(self.J):
                dispatch_prob = np.sum(dispatch_matrix[n,j] * pi)
                total_dispatches[n] += dispatch_prob
                if j not in district_atoms:
                    interdistrict[n] += dispatch_prob
        
        # Compute fractions
        interdistrict_fractions = interdistrict / (total_dispatches + 1e-10)
        
        return {
            'overall': np.mean(interdistrict_fractions),
            'by_unit': interdistrict_fractions,
            'max': np.max(interdistrict_fractions),
            'min': np.min(interdistrict_fractions),
            'total_interdistrict': np.sum(interdistrict),
            'total_dispatches': np.sum(total_dispatches)
        }
    
    def compute_queue_metrics(self, pi: np.ndarray) -> Dict[str, float]:
        """Compute queue-related performance metrics.
        
        Args:
            pi (numpy.ndarray): Steady state probabilities
            
        Returns:
            Dict[str, float]: Queue metrics
        """
        # Probability of all servers busy
        p_all_busy = sum(p for i, p in enumerate(pi) if bin(i).count('1') == self.N)
        
        # M/M/N queue metrics
        factorial_N = np.math.factorial(self.N)
        sum_term = sum((self.N * self.rho)**n / np.math.factorial(n) 
                      for n in range(self.N))
        p0 = 1 / (sum_term + (self.N * self.rho)**self.N / 
                  (factorial_N * (1 - self.rho)))
        
        # Queue length metrics
        L_q = (p0 * (self.lambda_rate/self.mu_rate)**self.N * self.rho) / \
              (factorial_N * (1 - self.rho)**2)
        W_q = L_q / self.lambda_rate  # Mean wait time in queue
        
        return {
            'p_all_busy': p_all_busy,
            'mean_queue_length': L_q,
            'mean_wait_time': W_q,
            'utilization': self.rho,
            'p0': p0,
            'server_utilization': 1 - p0,
            'probability_delay': p_all_busy
        }
    
    def compute_all_metrics(self, states: np.ndarray, pi: np.ndarray, 
                          T: np.ndarray, dispatch_probs: np.ndarray,
                          districts: List[List[int]], 
                          include_queue: bool = True) -> PerformanceMetrics:
        """Compute all system performance metrics.
        
        Args:
            states (numpy.ndarray): System states
            pi (numpy.ndarray): Steady state probabilities
            T (numpy.ndarray): Travel time matrix
            dispatch_probs (numpy.ndarray): Dispatch probabilities
            districts (List[List[int]]): District definitions
            include_queue (bool): Whether to include queue metrics
            
        Returns:
            PerformanceMetrics: All system performance metrics
        """
        # Compute workloads
        workloads = self.compute_workloads(states, pi)
        
        # Compute travel times
        travel_times = self.compute_travel_times(states, pi, T, dispatch_probs)
        
        # Compute interdistrict metrics
        interdistrict = self.compute_interdistrict_metrics(
            states, pi, districts, dispatch_probs
        )
        
        # Compute queue metrics if requested
        queue_metrics = None
        if include_queue:
            queue_metrics = self.compute_queue_metrics(pi)
            
        return PerformanceMetrics(
            workloads=workloads,
            travel_times=travel_times,
            interdistrict_fractions=interdistrict['by_unit'],
            queue_metrics=queue_metrics
        )