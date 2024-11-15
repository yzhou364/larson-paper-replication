import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.models.base_model import BaseHypercubeModel, ModelConfig
from src.core.transition_matrix import initialize_transition_matrix
from src.core.steady_state import SteadyStateCalculator
from src.core.performance import PerformanceAnalyzer

@dataclass
class ZeroLineModelResults:
    """Results container for zero-line capacity model."""
    steady_state_probs: np.ndarray
    workloads: np.ndarray
    travel_times: Dict[str, float]
    interdistrict_fractions: np.ndarray
    performance_summary: Dict[str, float]

class ZeroLineModel(BaseHypercubeModel):
    """Implementation of zero-line capacity hypercube model."""
    
    def __init__(self, config: ModelConfig):
        """Initialize zero-line capacity model.
        
        Args:
            config (ModelConfig): Model configuration
        """
        super().__init__(config)
        self.dispatch_probs = None
        
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
            return self.N  # Return artificial unit N+1
            
        # Compute mean travel times for available units
        travel_times = []
        for n in available_units:
            time = sum(self.L[n,i] * self.T[i,atom] for i in range(self.J))
            travel_times.append((n, time))
            
        # Return unit with minimum travel time
        return min(travel_times, key=lambda x: x[1])[0]
    
    def compute_dispatch_probabilities(self) -> np.ndarray:
        """Compute dispatch probability matrix.
        
        Returns:
            numpy.ndarray: Matrix of dispatch probabilities
        """
        dispatch_probs = np.zeros((self.N, self.J))
        
        for state_num in range(2**self.N):
            state = self.get_state_representation(state_num)
            prob = self.steady_state_probs[state_num]
            
            for atom in range(self.J):
                unit = self.get_optimal_dispatch(state, atom)
                if unit < self.N:  # Don't count artificial unit
                    dispatch_probs[unit, atom] += prob * self.f[atom]
                    
        return dispatch_probs
    
    def compute_performance_measures(self) -> Dict:
        """Compute performance measures for zero-line capacity system.
        
        Returns:
            Dict: Performance measures
        """
        if self.steady_state_probs is None:
            raise ValueError("Must run model before computing performance measures")
            
        # Get states and compute dispatch probabilities
        states = np.arange(2**self.N)
        self.dispatch_probs = self.compute_dispatch_probabilities()
        
        # Compute all metrics using performance analyzer
        metrics = self.performance_analyzer.compute_all_metrics(
            states=states,
            pi=self.steady_state_probs,
            T=self.T,
            dispatch_probs=self.dispatch_probs,
            districts=self.district_assignments,
            include_queue=False  # Zero-line capacity model
        )
        
        return {
            'workloads': metrics.workloads,
            'travel_times': metrics.travel_times,
            'interdistrict_fractions': metrics.interdistrict_fractions,
            'performance_summary': self.get_performance_summary()
        }
    
    def run(self) -> ZeroLineModelResults:
        """Run zero-line capacity model and return results.
        
        Returns:
            ZeroLineModelResults: Model results
        """
        # Initialize system
        self.initialize_system()
        
        # Compute steady state probabilities
        self.compute_steady_state(method='direct')
        
        # Compute performance measures
        measures = self.compute_performance_measures()
        
        return ZeroLineModelResults(
            steady_state_probs=self.steady_state_probs,
            workloads=measures['workloads'],
            travel_times=measures['travel_times'],
            interdistrict_fractions=measures['interdistrict_fractions'],
            performance_summary=measures['performance_summary']
        )
    
    def analyze_dispatch_patterns(self) -> Dict:
        """Analyze dispatch patterns and preferences.
        
        Returns:
            Dict: Dispatch pattern analysis
        """
        if self.dispatch_probs is None:
            raise ValueError("Must run model before analyzing dispatch patterns")
            
        analysis = {}
        
        # Primary response patterns
        primary_responses = np.argmax(self.dispatch_probs, axis=0)
        analysis['primary_units'] = primary_responses
        
        # Response frequencies
        analysis['response_frequencies'] = np.sum(self.dispatch_probs, axis=1)
        
        # Dispatch preferences
        preferences = {}
        for atom in range(self.J):
            # Sort units by dispatch probability for this atom
            prefs = np.argsort(-self.dispatch_probs[:, atom])
            preferences[atom] = prefs.tolist()
        analysis['dispatch_preferences'] = preferences
        
        # Workload distribution analysis
        analysis['workload_distribution'] = {
            'by_district': [
                sum(self.dispatch_probs[n, atom] 
                    for atom in self.district_assignments[n])
                for n in range(self.N)
            ],
            'interdistrict_fraction': [
                1 - sum(self.dispatch_probs[n, atom] 
                       for atom in self.district_assignments[n])
                for n in range(self.N)
            ]
        }
        
        return analysis
    
    def get_coverage_metrics(self, target_time: float) -> Dict:
        """Compute coverage metrics for target response time.
        
        Args:
            target_time (float): Target response time
            
        Returns:
            Dict: Coverage metrics
        """
        coverage = np.zeros(self.J)
        
        for j in range(self.J):
            # Probability of response within target time
            coverage[j] = sum(
                self.dispatch_probs[n,j] 
                for n in range(self.N)
                if np.mean([self.T[i,j] for i in self.district_assignments[n]]) <= target_time
            )
            
        return {
            'coverage_by_atom': coverage,
            'mean_coverage': np.mean(coverage),
            'min_coverage': np.min(coverage),
            'max_coverage': np.max(coverage),
            'atoms_covered': np.sum(coverage >= 0.90)  # Number of atoms with 90% coverage
        }
    
    def validate_results(self) -> bool:
        """Validate model results.
        
        Returns:
            bool: True if results are valid
        """
        if self.steady_state_probs is None:
            return False
            
        # Check probability sum
        if not np.isclose(np.sum(self.steady_state_probs), 1.0):
            return False
            
        # Check dispatch probability consistency
        if self.dispatch_probs is not None:
            row_sums = np.sum(self.dispatch_probs, axis=1)
            if not np.allclose(row_sums / np.sum(row_sums), 
                             self.performance_analyzer.compute_workloads(
                                 np.arange(2**self.N), 
                                 self.steady_state_probs
                             ) / np.sum(self.performance_analyzer.compute_workloads(
                                 np.arange(2**self.N),
                                 self.steady_state_probs
                             ))):
                return False
                
        return True