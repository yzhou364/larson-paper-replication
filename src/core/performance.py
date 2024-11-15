import numpy as np
from typing import List, Dict, Optional, Tuple

class PerformanceMetrics:
    """Compute and analyze system performance metrics."""
    
    @staticmethod
    def compute_workload(N: int, states: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Compute the workload (fraction of time busy) for each unit.
        
        Args:
            N (int): Number of units
            states (numpy.ndarray): Binary state representations
            pi (numpy.ndarray): Steady state probabilities
            
        Returns:
            numpy.ndarray: Workload for each unit
        """
        workloads = np.zeros(N)
        for i, state in enumerate(states):
            state_binary = format(state, f'0{N}b')
            for j in range(N):
                if state_binary[j] == '1':
                    workloads[j] += pi[i]
        return workloads

    @staticmethod
    def compute_workload_imbalance(workloads: np.ndarray) -> Dict[str, float]:
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

    @staticmethod
    def compute_travel_times(states: np.ndarray, pi: np.ndarray, T: np.ndarray, 
                           dispatch_probs: np.ndarray) -> Dict[str, float]:
        """Compute travel time metrics.
        
        Args:
            states (numpy.ndarray): System states
            pi (numpy.ndarray): Steady state probabilities
            T (numpy.ndarray): Travel time matrix
            dispatch_probs (numpy.ndarray): Dispatch probabilities
            
        Returns:
            Dict[str, float]: Travel time metrics
        """
        avg_travel_time = np.sum(T * dispatch_probs)
        weighted_times = T * dispatch_probs * pi[:, np.newaxis]
        
        return {
            'average': avg_travel_time,
            'weighted': np.sum(weighted_times),
            'max': np.max(T * (dispatch_probs > 0)),
            'std': np.std(T * (dispatch_probs > 0)),
            'by_unit': np.sum(weighted_times, axis=1),
            'by_atom': np.sum(weighted_times, axis=0)
        }

    @staticmethod
    def compute_interdistrict_fraction(states: np.ndarray, pi: np.ndarray, 
                                     districts: List[List[int]], 
                                     dispatch_matrix: np.ndarray) -> Dict[str, float]:
        """Compute interdistrict dispatch metrics.
        
        Args:
            states (numpy.ndarray): System states
            pi (numpy.ndarray): Steady state probabilities
            districts (List[List[int]]): District definitions
            dispatch_matrix (numpy.ndarray): Dispatch probability matrix
            
        Returns:
            Dict[str, float]: Interdistrict metrics
        """
        N = len(districts)
        interdistrict = np.zeros(N)
        total_dispatches = np.zeros(N)
        
        # Compute interdistrict dispatches for each unit
        for n in range(N):
            district_atoms = set(districts[n])
            for j in range(dispatch_matrix.shape[1]):
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

    @staticmethod
    def compute_queue_metrics(N: int, lambda_rate: float, mu_rate: float, 
                            pi: np.ndarray) -> Dict[str, float]:
        """Compute queue performance metrics.
        
        Args:
            N (int): Number of units
            lambda_rate (float): Arrival rate
            mu_rate (float): Service rate
            pi (numpy.ndarray): Steady state probabilities
            
        Returns:
            Dict[str, float]: Queue metrics
        """
        rho = lambda_rate / (N * mu_rate)
        p_all_busy = sum(p for i, p in enumerate(pi) if bin(i).count('1') == N)
        
        # M/M/N queue metrics
        factorial_N = np.math.factorial(N)
        sum_term = sum((N * rho)**n / np.math.factorial(n) for n in range(N))
        p0 = 1 / (sum_term + (N * rho)**N / (factorial_N * (1 - rho)))
        
        # Queue length metrics
        L_q = (p0 * (lambda_rate/mu_rate)**N * rho) / (factorial_N * (1 - rho)**2)
        W_q = L_q / lambda_rate  # Mean wait time in queue
        
        return {
            'p_all_busy': p_all_busy,
            'mean_queue_length': L_q,
            'mean_wait_time': W_q,
            'utilization': rho,
            'p0': p0,
            'server_utilization': 1 - p0,
            'probability_delay': p_all_busy,
            'mean_number_busy': sum(bin(i).count('1') * p for i, p in enumerate(pi))
        }

    @staticmethod
    def compute_regional_metrics(pi: np.ndarray, T: np.ndarray, f: np.ndarray,
                               dispatch_probs: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute region-specific performance metrics.
        
        Args:
            pi (numpy.ndarray): Steady state probabilities
            T (numpy.ndarray): Travel time matrix
            f (numpy.ndarray): Demand distribution
            dispatch_probs (numpy.ndarray): Dispatch probabilities
            
        Returns:
            Dict[str, np.ndarray]: Regional metrics
        """
        J = len(f)
        region_metrics = {
            'mean_response_time': np.zeros(J),
            'workload_density': np.zeros(J),
            'dispatch_frequency': np.zeros(J),
            'coverage': np.zeros(J)
        }
        
        # Compute metrics for each region
        for j in range(J):
            # Mean response time to region j
            region_metrics['mean_response_time'][j] = np.sum(T[:,j] * dispatch_probs[:,j])
            
            # Workload density in region j
            region_metrics['workload_density'][j] = f[j]
            
            # Dispatch frequency to region j
            region_metrics['dispatch_frequency'][j] = np.sum(dispatch_probs[:,j])
            
            # Coverage (probability of available unit within target time)
            coverage_time = np.mean(T[:,j])  # Use mean travel time as target
            region_metrics['coverage'][j] = np.sum(
                dispatch_probs[:,j] * (T[:,j] <= coverage_time)
            )
            
        return region_metrics