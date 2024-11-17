import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.models.base_model import BaseHypercubeModel, ModelConfig
from src.core.transition_matrix import initialize_transition_matrix
from src.core.steady_state import SteadyStateCalculator
from src.core.performance import PerformanceMetrics

@dataclass
class InfiniteLineModelResults:
    """Results container for infinite-line capacity model."""
    steady_state_probs: np.ndarray
    queue_probs: np.ndarray
    workloads: np.ndarray
    travel_times: Dict[str, float]
    interdistrict_fractions: np.ndarray
    queue_metrics: Dict[str, float]
    performance_summary: Dict[str, float]

class InfiniteLineModel(BaseHypercubeModel):
    """Implementation of infinite-line capacity hypercube model."""
    
    def __init__(self, config: ModelConfig):
        """Initialize infinite-line capacity model.
        
        Args:
            config (ModelConfig): Model configuration
        """
        super().__init__(config)
        self.dispatch_probs = None
        self.queue_probs = None
        
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
            # For infinite-line model, randomly assign one of the busy units
            return np.random.randint(0, self.N)
            
        # Compute mean travel times for available units
        travel_times = []
        for n in available_units:
            time = sum(self.L[n,i] * self.T[i,atom] for i in range(self.J))
            travel_times.append((n, time))
            
        # Return unit with minimum travel time
        return min(travel_times, key=lambda x: x[1])[0]
    
    def compute_queue_probabilities(self):
        """Compute queue state probabilities."""
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
    
    def compute_dispatch_probabilities(self) -> np.ndarray:
        """Compute dispatch probability matrix.
        
        Returns:
            numpy.ndarray: Matrix of dispatch probabilities
        """
        dispatch_probs = np.zeros((self.N, self.J))
        
        # Non-queued dispatches
        for state_num in range(2**self.N):
            state = self.get_state_representation(state_num)
            prob = self.steady_state_probs[state_num]
            
            for atom in range(self.J):
                unit = self.get_optimal_dispatch(state, atom)
                dispatch_probs[unit, atom] += prob * self.f[atom]
        
        # Queued dispatches (equal probability to all units)
        if self.queue_probs is not None:
            queue_prob = sum(self.queue_probs)
            for atom in range(self.J):
                dispatch_probs[:, atom] += (queue_prob * self.f[atom] / self.N)
                    
        return dispatch_probs
    
    def _compute_queue_metrics(self) -> Dict:
        """Compute queue-specific performance metrics."""
        queue_prob = sum(self.queue_probs)
        expected_queue_length = sum(k * p for k, p in enumerate(self.queue_probs))
        
        return {
            'probability_queue': queue_prob,
            'expected_queue_length': expected_queue_length,
            'expected_wait_time': expected_queue_length / self.lambda_rate,
            'total_delay': sum(self.queue_probs) / self.mu_rate,
            'queue_distribution': {
                k: p for k, p in enumerate(self.queue_probs) if p > 1e-10
            }
        }
    
    def _compute_queue_travel_time(self) -> float:
        """Compute mean travel time for queued calls."""
        queue_time = 0
        for i in range(self.J):
            for j in range(self.J):
                queue_time += self.f[i] * self.f[j] * self.T[i,j]
        return queue_time
    
    def _adjust_metrics_for_queue(self, metrics: PerformanceMetrics, 
                                queue_metrics: Dict) -> PerformanceMetrics:
        """Adjust performance metrics for queuing effects."""
        queue_prob = queue_metrics['probability_queue']
        queue_travel_time = self._compute_queue_travel_time()
        
        # Adjust workloads
        adjusted_workloads = metrics.workloads + (queue_prob / self.N)
        
        # Adjust travel times
        adjusted_travel_times = {}
        for metric, value in metrics.travel_times.items():
            if isinstance(value, (int, float)):
                adjusted_travel_times[metric] = (
                    (1 - queue_prob) * value + queue_prob * queue_travel_time
                )
            elif isinstance(value, np.ndarray):
                adjusted_travel_times[metric] = (
                    (1 - queue_prob) * value + queue_prob * queue_travel_time
                )
        
        return PerformanceMetrics(
            workloads=adjusted_workloads,
            travel_times=adjusted_travel_times,
            interdistrict_fractions=metrics.interdistrict_fractions,
            queue_metrics=queue_metrics
        )
    
    def compute_performance_measures(self) -> Dict:
        """Compute performance measures for infinite-line capacity system.
        
        Returns:
            Dict: Performance measures
        """
        if self.steady_state_probs is None or self.queue_probs is None:
            raise ValueError("Must run model before computing performance measures")
            
        # Get states and compute dispatch probabilities
        states = np.arange(2**self.N)
        self.dispatch_probs = self.compute_dispatch_probabilities()
        
        # Compute queue metrics
        queue_metrics = self._compute_queue_metrics()
        
        # Compute basic metrics using performance analyzer
        metrics = self.performance_analyzer.compute_all_metrics(
            states=states,
            pi=self.steady_state_probs,
            T=self.T,
            dispatch_probs=self.dispatch_probs,
            districts=self.district_assignments,
            include_queue=True
        )
        
        # Adjust metrics for queuing
        adjusted_metrics = self._adjust_metrics_for_queue(metrics, queue_metrics)
        
        return {
            'workloads': adjusted_metrics.workloads,
            'travel_times': adjusted_metrics.travel_times,
            'interdistrict_fractions': adjusted_metrics.interdistrict_fractions,
            'queue_metrics': queue_metrics,
            'performance_summary': self.get_performance_summary()
        }
    
    def run(self) -> InfiniteLineModelResults:
        """Run infinite-line capacity model and return results.
        
        Returns:
            InfiniteLineModelResults: Model results
        """
        # Initialize system
        self.initialize_system()
        
        # Compute steady state probabilities
        self.compute_steady_state(method='direct')
        
        # Compute queue probabilities
        self.compute_queue_probabilities()
        
        # Compute performance measures
        measures = self.compute_performance_measures()
        
        return InfiniteLineModelResults(
            steady_state_probs=self.steady_state_probs,
            queue_probs=np.array(self.queue_probs),
            workloads=measures['workloads'],
            travel_times=measures['travel_times'],
            interdistrict_fractions=measures['interdistrict_fractions'],
            queue_metrics=measures['queue_metrics'],
            performance_summary=measures['performance_summary']
        )
    
    def get_queue_length_distribution(self) -> Dict[int, float]:
        """Get probability distribution of queue lengths.
        
        Returns:
            Dict[int, float]: Queue length distribution
        """
        if self.queue_probs is None:
            raise ValueError("Must run model before getting queue distribution")
            
        return {k: p for k, p in enumerate(self.queue_probs) if p > 1e-10}
    
    def get_expected_queue_length(self) -> float:
        """Calculate expected queue length.
        
        Returns:
            float: Expected queue length
        """
        if self.queue_probs is None:
            raise ValueError("Must run model before calculating queue length")
            
        return sum(k * p for k, p in enumerate(self.queue_probs))
    
    def get_expected_system_time(self) -> float:
        """Calculate expected total time in system (wait + service).
        
        Returns:
            float: Expected system time
        """
        queue_metrics = self._compute_queue_metrics()
        service_time = 1.0 / self.mu_rate
        return queue_metrics['expected_wait_time'] + service_time