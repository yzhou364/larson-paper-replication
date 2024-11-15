"""
Sensitivity analysis for hypercube queuing model parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
from scipy.stats import norm
import logging
from itertools import product

@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    param_ranges: Dict[str, Tuple[float, float]]
    num_samples: int = 100
    use_latin_hypercube: bool = True
    output_metrics: List[str] = None
    confidence_level: float = 0.95

class SensitivityAnalyzer:
    """Analyzes system sensitivity to parameter variations."""
    
    def __init__(self, model_runner: Callable, config: Optional[SensitivityConfig] = None):
        """Initialize sensitivity analyzer.
        
        Args:
            model_runner (Callable): Function to run model with parameters
            config (Optional[SensitivityConfig]): Analysis configuration
        """
        self.model_runner = model_runner
        self.config = config or SensitivityConfig(param_ranges={})
        self.logger = logging.getLogger(__name__)
        
        if self.config.output_metrics is None:
            self.config.output_metrics = [
                'workload_balance',
                'response_time',
                'coverage',
                'queue_length'
            ]
            
    def analyze_parameter_sensitivity(self) -> Dict:
        """Perform comprehensive parameter sensitivity analysis.
        
        Returns:
            Dict: Sensitivity analysis results
        """
        # Generate parameter samples
        samples = self._generate_samples()
        
        # Run model for each sample
        results = []
        for sample in samples:
            try:
                result = self.model_runner(**sample)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Model run failed for sample {sample}: {str(e)}")
                
        # Analyze sensitivity
        analysis = {
            'local': self._analyze_local_sensitivity(samples, results),
            'global': self._analyze_global_sensitivity(samples, results),
            'interactions': self._analyze_parameter_interactions(samples, results),
            'thresholds': self._identify_thresholds(samples, results),
            'robustness': self._analyze_robustness(samples, results)
        }
        
        return analysis
    
    def _generate_samples(self) -> List[Dict]:
        """Generate parameter samples for analysis.
        
        Returns:
            List[Dict]: Parameter samples
        """
        if self.config.use_latin_hypercube:
            return self._latin_hypercube_sampling()
        else:
            return self._random_sampling()
            
    def _latin_hypercube_sampling(self) -> List[Dict]:
        """Generate Latin Hypercube samples.
        
        Returns:
            List[Dict]: Parameter samples
        """
        n_params = len(self.config.param_ranges)
        n_samples = self.config.num_samples
        
        # Generate Latin Hypercube samples
        result = []
        
        # Create the Latin Hypercube grid
        cut = np.linspace(0, 1, n_samples + 1)
        grid = np.random.uniform(low=cut[:-1], high=cut[1:], size=(n_params, n_samples))
        
        # Shuffle each row
        for i in range(n_params):
            np.random.shuffle(grid[i,:])
            
        # Convert to parameter values
        param_names = list(self.config.param_ranges.keys())
        for i in range(n_samples):
            sample = {}
            for j, param in enumerate(param_names):
                low, high = self.config.param_ranges[param]
                sample[param] = low + grid[j,i] * (high - low)
            result.append(sample)
            
        return result
    
    def _random_sampling(self) -> List[Dict]:
        """Generate random parameter samples.
        
        Returns:
            List[Dict]: Parameter samples
        """
        result = []
        for _ in range(self.config.num_samples):
            sample = {}
            for param, (low, high) in self.config.param_ranges.items():
                sample[param] = np.random.uniform(low, high)
            result.append(sample)
            
        return result
    
    def _analyze_local_sensitivity(self, samples: List[Dict],
                                results: List[Dict]) -> Dict:
        """Analyze local parameter sensitivity.
        
        Args:
            samples (List[Dict]): Parameter samples
            results (List[Dict]): Model results
            
        Returns:
            Dict: Local sensitivity analysis
        """
        sensitivities = {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(samples)
        df_results = pd.DataFrame([
            {metric: result[metric] for metric in self.config.output_metrics}
            for result in results
        ])
        
        # Compute partial derivatives
        for param in self.config.param_ranges.keys():
            param_sensitivities = {}
            for metric in self.config.output_metrics:
                # Linear regression for sensitivity
                slope = np.polyfit(df[param], df_results[metric], 1)[0]
                param_sensitivities[metric] = {
                    'slope': slope,
                    'normalized': slope * (
                        np.std(df[param]) / np.std(df_results[metric])
                    )
                }
            sensitivities[param] = param_sensitivities
            
        return sensitivities
    
    def _analyze_global_sensitivity(self, samples: List[Dict],
                                 results: List[Dict]) -> Dict:
        """Analyze global parameter sensitivity.
        
        Args:
            samples (List[Dict]): Parameter samples
            results (List[Dict]): Model results
            
        Returns:
            Dict: Global sensitivity analysis
        """
        # Convert to arrays
        X = np.array([[s[p] for p in self.config.param_ranges.keys()] 
                     for s in samples])
        Y = np.array([[r[m] for m in self.config.output_metrics] 
                     for r in results])
        
        # Compute main effects (first-order sensitivity)
        main_effects = {}
        for i, param in enumerate(self.config.param_ranges.keys()):
            effects = {}
            for j, metric in enumerate(self.config.output_metrics):
                effects[metric] = self._compute_sobol_index(X[:,i], Y[:,j])
            main_effects[param] = effects
            
        # Compute total effects
        total_effects = self._compute_total_effects(X, Y)
        
        return {
            'main_effects': main_effects,
            'total_effects': total_effects
        }
    
    def _analyze_parameter_interactions(self, samples: List[Dict],
                                    results: List[Dict]) -> Dict:
        """Analyze parameter interactions.
        
        Args:
            samples (List[Dict]): Parameter samples
            results (List[Dict]): Model results
            
        Returns:
            Dict: Parameter interaction analysis
        """
        interactions = {}
        params = list(self.config.param_ranges.keys())
        
        # Analyze pairwise interactions
        for i, param1 in enumerate(params):
            for j, param2 in enumerate(params[i+1:], i+1):
                interaction = self._analyze_pairwise_interaction(
                    samples, results, param1, param2
                )
                interactions[f"{param1}_{param2}"] = interaction
                
        return interactions
    
    def _identify_thresholds(self, samples: List[Dict],
                          results: List[Dict]) -> Dict:
        """Identify parameter thresholds.
        
        Args:
            samples (List[Dict]): Parameter samples
            results (List[Dict]): Model results
            
        Returns:
            Dict: Parameter thresholds
        """
        thresholds = {}
        
        for param in self.config.param_ranges.keys():
            param_thresholds = {}
            param_values = [s[param] for s in samples]
            
            for metric in self.config.output_metrics:
                metric_values = [r[metric] for r in results]
                
                # Find potential threshold points
                threshold = self._find_metric_threshold(
                    param_values, metric_values
                )
                param_thresholds[metric] = threshold
                
            thresholds[param] = param_thresholds
            
        return thresholds
    
    def _analyze_robustness(self, samples: List[Dict],
                         results: List[Dict]) -> Dict:
        """Analyze system robustness to parameter variations.
        
        Args:
            samples (List[Dict]): Parameter samples
            results (List[Dict]): Model results
            
        Returns:
            Dict: Robustness analysis
        """
        robustness = {}
        
        for metric in self.config.output_metrics:
            metric_values = [r[metric] for r in results]
            
            # Compute robustness metrics
            robustness[metric] = {
                'cv': np.std(metric_values) / np.mean(metric_values),
                'range_ratio': (max(metric_values) - min(metric_values)) / np.mean(metric_values),
                'reliability': np.mean([
                    1 if v >= np.mean(metric_values) * 0.9 else 0
                    for v in metric_values
                ])
            }
            
        return robustness
    
    def _compute_sobol_index(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute first-order Sobol sensitivity index.
        
        Args:
            X (numpy.ndarray): Parameter values
            Y (numpy.ndarray): Output values
            
        Returns:
            float: Sobol index
        """
        # Estimate first-order Sobol index using correlation ratio
        bins = min(30, len(X) // 10)
        digitized = np.digitize(X, np.linspace(X.min(), X.max(), bins))
        bin_means = np.array([Y[digitized == i].mean() for i in range(1, bins+1)])
        bin_var = np.array([Y[digitized == i].var() for i in range(1, bins+1)])
        
        # Compute index
        total_var = Y.var()
        if total_var == 0:
            return 0
            
        return (bin_means.var() * np.mean(bin_var)) / total_var
    
    def _compute_total_effects(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """Compute total effect sensitivity indices.
        
        Args:
            X (numpy.ndarray): Parameter values
            Y (numpy.ndarray): Output values
            
        Returns:
            Dict: Total effect indices
        """
        total_effects = {}
        
        for j, metric in enumerate(self.config.output_metrics):
            effects = {}
            for i, param in enumerate(self.config.param_ranges.keys()):
                # Estimate total effect using variance decomposition
                other_params = list(range(X.shape[1]))
                other_params.remove(i)
                
                if other_params:
                    cond_var = np.array([
                        np.var(Y[X[:,other_params] == x[:,other_params], j])
                        for x in X
                    ])
                    effects[param] = np.mean(cond_var) / np.var(Y[:,j])
                else:
                    effects[param] = 1.0
                    
            total_effects[metric] = effects
            
        return total_effects
    
    def _analyze_pairwise_interaction(self, samples: List[Dict],
                                   results: List[Dict],
                                   param1: str, param2: str) -> Dict:
        """Analyze interaction between two parameters.
        
        Args:
            samples (List[Dict]): Parameter samples
            results (List[Dict]): Model results
            param1 (str): First parameter
            param2 (str): Second parameter
            
        Returns:
            Dict: Interaction analysis
        """
        interaction = {}
        
        # Convert to arrays
        x1 = np.array([s[param1] for s in samples])
        x2 = np.array([s[param2] for s in samples])
        
        for metric in self.config.output_metrics:
            y = np.array([r[metric] for r in results])
            
            # Compute interaction metrics
            interaction[metric] = {
                'correlation': np.corrcoef(x1 * x2, y)[0,1],
                'synergy': self._compute_synergy(x1, x2, y)
            }
            
        return interaction
    
    def _find_metric_threshold(self, param_values: List[float],
                            metric_values: List[float]) -> Dict:
        """Find threshold points in metric response.
        
        Args:
            param_values (List[float]): Parameter values
            metric_values (List[float]): Metric values
            
        Returns:
            Dict: Threshold analysis
        """
        # Sort by parameter value
        sorted_indices = np.argsort(param_values)
        x = np.array(param_values)[sorted_indices]
        y = np.array(metric_values)[sorted_indices]
        
        # Compute changes
        changes = np.diff(y) / np.diff(x)
        
        # Find potential threshold points
        threshold_points = []
        for i in range(1, len(changes)):
            if abs(changes[i] - changes[i-1]) > np.std(changes):
                threshold_points.append({
                    'value': x[i],
                    'change': changes[i] - changes[i-1]
                })
                
        return {
            'points': threshold_points,
            'gradual_region': [np.min(x), np.max(x)],
            'stability': np.std(changes)
        }
        
    def _compute_synergy(self, x1: np.ndarray, x2: np.ndarray,
                       y: np.ndarray) -> float:
        """Compute synergy between parameters.
        
        Args:
            x1 (numpy.ndarray): First parameter values
            x2 (numpy.ndarray): Second parameter values
            y (numpy.ndarray): Output values
            
        Returns:
            float: Synergy measure
        """
        # Compute individual effects
        effect1 = np.polyfit(x1, y, 1)[0]
        effect2 = np.polyfit(x2, y, 1)[0]
        
        # Compute joint effect
        combined = x1 * x2
        joint_effect = np.polyfit(combined, y, 1)[0]
        
        # Compute synergy as difference from additive effects
        return joint_effect - (effect1 + effect2)