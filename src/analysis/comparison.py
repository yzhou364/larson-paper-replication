"""
Comparison analysis for hypercube model configurations and results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error
import logging

@dataclass
class ComparisonConfig:
    """Configuration for comparison analysis."""
    significance_level: float = 0.05
    use_bootstrap: bool = True
    num_bootstrap: int = 1000
    metrics_of_interest: List[str] = None

class SystemComparator:
    """Compares different hypercube system configurations."""
    
    def __init__(self, config: Optional[ComparisonConfig] = None):
        """Initialize system comparator.
        
        Args:
            config (Optional[ComparisonConfig]): Comparison configuration
        """
        self.config = config or ComparisonConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.metrics_of_interest is None:
            self.config.metrics_of_interest = [
                'workload_balance',
                'response_time',
                'coverage',
                'queue_performance'
            ]
            
    def compare_queue_types(self, zero_line: Dict, infinite_line: Dict) -> Dict:
        """Compare zero-line and infinite-line capacity systems.
        
        Args:
            zero_line (Dict): Zero-line capacity results
            infinite_line (Dict): Infinite-line capacity results
            
        Returns:
            Dict: Comparison results
        """
        comparison = {}
        
        # Get the rho values (system utilization levels)
        rho_values = sorted(zero_line.keys())
        
        # Compare workload distributions
        workload_comparison = []
        for rho in rho_values:
            workload_comparison.append(self._compare_distributions(
                zero_line[rho]['workloads'],
                infinite_line[rho]['workloads'],
                'Workload'
            ))
        comparison['workload'] = {
            'by_rho': workload_comparison,
            'average': {
                'relative_change': np.mean([comp['basic_stats']['relative_change'] 
                                        for comp in workload_comparison])
            }
        }
        
        # Compare response times
        travel_time_comparison = []
        for rho in rho_values:
            travel_time_comparison.append(self._compare_distributions(
                [zero_line[rho]['travel_times']['average']],
                [infinite_line[rho]['travel_times']['average']],
                'Response Time'
            ))
        comparison['response_time'] = {
            'by_rho': travel_time_comparison,
            'average': {
                'relative_change': np.mean([comp['basic_stats']['relative_change'] 
                                        for comp in travel_time_comparison])
            }
        }
        
        # Compare interdistrict fractions
        interdistrict_comparison = []
        for rho in rho_values:
            interdistrict_comparison.append(self._compare_distributions(
                zero_line[rho]['interdistrict_fraction'],
                infinite_line[rho]['interdistrict_fraction'],
                'Interdistrict'
            ))
        comparison['interdistrict'] = {
            'by_rho': interdistrict_comparison,
            'average': {
                'relative_change': np.mean([comp['basic_stats']['relative_change'] 
                                        for comp in interdistrict_comparison])
            }
        }
        
        # Add queue metrics analysis (infinite-line only)
        queue_metrics = []
        for rho in rho_values:
            if 'queue_metrics' in infinite_line[rho]:
                metrics = infinite_line[rho]['queue_metrics']
                queue_metrics.append({
                    'rho': rho,
                    'expected_queue_length': metrics['expected_queue_length'],
                    'expected_wait_time': metrics['expected_wait_time'],
                    'probability_queue': metrics['probability_queue'],
                    'total_delay': metrics['total_delay']
                })
        
        comparison['queue_performance'] = {
            'by_rho': queue_metrics,
            'average': {
                'expected_queue_length': np.mean([qm['expected_queue_length'] 
                                                for qm in queue_metrics]),
                'expected_wait_time': np.mean([qm['expected_wait_time'] 
                                            for qm in queue_metrics]),
                'probability_queue': np.mean([qm['probability_queue'] 
                                            for qm in queue_metrics])
            } if queue_metrics else {}
        }
        
        return comparison

    def compare_configurations(self, configs: List[Dict], 
                             results: List[Dict]) -> Dict:
        """Compare multiple system configurations.
        
        Args:
            configs (List[Dict]): System configurations
            results (List[Dict]): Corresponding results
            
        Returns:
            Dict: Configuration comparison results
        """
        comparison = {
            'pairwise': self._perform_pairwise_comparison(results),
            'ranking': self._rank_configurations(results),
            'trade_offs': self._analyze_trade_offs(configs, results)
        }
        
        # Statistical significance testing
        if len(configs) > 1:
            comparison['significance'] = self._test_significance(results)
            
        return comparison
    
    def compare_policies(self, policies: List[Dict],
                        metrics: List[Dict]) -> Dict:
        """Compare different dispatch policies.
        
        Args:
            policies (List[Dict]): Policy configurations
            metrics (List[Dict]): Performance metrics
            
        Returns:
            Dict: Policy comparison results
        """
        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame([
            {**policy, **metric}
            for policy, metric in zip(policies, metrics)
        ])
        
        comparison = {
            'performance_summary': self._summarize_policy_performance(df),
            'statistical_tests': self._test_policy_differences(df),
            'recommendations': self._generate_policy_recommendations(df)
        }
        
        return comparison
    
    def _compare_distributions(self, dist1: np.ndarray, dist2: np.ndarray,
                            metric_name: str) -> Dict:
        """Compare two distributions statistically.
        
        Args:
            dist1 (numpy.ndarray): First distribution
            dist2 (numpy.ndarray): Second distribution
            metric_name (str): Name of metric being compared
            
        Returns:
            Dict: Distribution comparison results
        """
        # Basic statistics
        basic_stats = {
            'mean_difference': np.mean(dist2) - np.mean(dist1),
            'std_difference': np.std(dist2) - np.std(dist1),
            'relative_change': (np.mean(dist2) - np.mean(dist1)) / np.mean(dist1)
        }
        
        # Statistical tests
        try:
            # t-test for means
            t_stat, p_value = stats.ttest_ind(dist1, dist2)
            statistical_tests = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.config.significance_level
            }
            
            # KS test for distribution equality
            ks_stat, ks_p = stats.ks_2samp(dist1, dist2)
            statistical_tests['ks_statistic'] = ks_stat
            statistical_tests['ks_p_value'] = ks_p
            
        except Exception as e:
            self.logger.warning(f"Statistical tests failed for {metric_name}: {str(e)}")
            statistical_tests = None
            
        # Bootstrap analysis if configured
        if self.config.use_bootstrap:
            bootstrap_results = self._bootstrap_comparison(
                dist1, dist2, self.config.num_bootstrap
            )
        else:
            bootstrap_results = None
            
        return {
            'metric': metric_name,
            'basic_stats': basic_stats,
            'statistical_tests': statistical_tests,
            'bootstrap_results': bootstrap_results
        }
    def _compare_service_levels(self, system1: Dict, system2: Dict) -> Dict:
        """Compare service levels between systems.
        
        Args:
            system1 (Dict): First system results
            system2 (Dict): Second system results
            
        Returns:
            Dict: Service level comparison
        """
        metrics = {}
        for metric in ['coverage', 'response_time', 'workload_balance']:
            if metric in system1 and metric in system2:
                metrics[metric] = {
                    'absolute_difference': system2[metric] - system1[metric],
                    'relative_difference': (system2[metric] - system1[metric]) / system1[metric],
                    'improvement': system2[metric] > system1[metric]
                }
                
        return metrics
    
    def _compare_efficiency(self, system1: Dict, system2: Dict) -> Dict:
        """Compare system efficiency metrics.
        
        Args:
            system1 (Dict): First system results
            system2 (Dict): Second system results
            
        Returns:
            Dict: Efficiency comparison
        """
        return {
            'resource_utilization': self._compare_utilization(system1, system2),
            'cost_effectiveness': self._compare_costs(system1, system2),
            'operational_efficiency': self._compare_operations(system1, system2)
        }
    
    def _bootstrap_comparison(self, data1: np.ndarray, 
                            data2: np.ndarray,
                            n_bootstrap: int) -> Dict:
        """Perform bootstrap comparison analysis.
        
        Args:
            data1 (numpy.ndarray): First dataset
            data2 (numpy.ndarray): Second dataset
            n_bootstrap (int): Number of bootstrap samples
            
        Returns:
            Dict: Bootstrap analysis results
        """
        differences = []
        for _ in range(n_bootstrap):
            # Generate bootstrap samples
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            
            # Compute difference in means
            differences.append(np.mean(sample2) - np.mean(sample1))
            
        differences = np.array(differences)
        
        return {
            'mean_difference': np.mean(differences),
            'confidence_interval': (
                np.percentile(differences, 2.5),
                np.percentile(differences, 97.5)
            ),
            'probability_improvement': np.mean(differences > 0)
        }
    
    def _perform_pairwise_comparison(self, results: List[Dict]) -> pd.DataFrame:
        """Perform pairwise comparison of results.
        
        Args:
            results (List[Dict]): List of results to compare
            
        Returns:
            pandas.DataFrame: Pairwise comparison matrix
        """
        n = len(results)
        metrics = self.config.metrics_of_interest
        comparison_matrix = np.zeros((n, n, len(metrics)))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    for k, metric in enumerate(metrics):
                        if metric in results[i] and metric in results[j]:
                            comparison_matrix[i,j,k] = (
                                results[j][metric] - results[i][metric]
                            )
                            
        return pd.Panel(comparison_matrix, 
                       items=range(n),
                       major_axis=range(n),
                       minor_axis=metrics).to_frame()
    
    def _rank_configurations(self, results: List[Dict]) -> Dict:
        """Rank configurations based on performance metrics.
        
        Args:
            results (List[Dict]): Results to rank
            
        Returns:
            Dict: Configuration rankings
        """
        rankings = {}
        for metric in self.config.metrics_of_interest:
            values = [result.get(metric) for result in results]
            if any(v is None for v in values):
                continue
                
            # Higher values are better
            rankings[metric] = np.argsort(values)[::-1]
            
        # Compute overall ranking using weighted sum
        weights = np.ones(len(self.config.metrics_of_interest))
        overall_scores = np.zeros(len(results))
        
        for i, result in enumerate(results):
            score = 0
            for j, metric in enumerate(self.config.metrics_of_interest):
                if metric in result:
                    score += weights[j] * result[metric]
            overall_scores[i] = score
            
        rankings['overall'] = np.argsort(overall_scores)[::-1]
        
        return rankings
    
    def _analyze_trade_offs(self, configs: List[Dict],
                          results: List[Dict]) -> Dict:
        """Analyze trade-offs between different configurations.
        
        Args:
            configs (List[Dict]): System configurations
            results (List[Dict]): Performance results
            
        Returns:
            Dict: Trade-off analysis
        """
        trade_offs = {}
        metrics = self.config.metrics_of_interest
        
        # Analyze pairwise trade-offs
        for i, metric1 in enumerate(metrics):
            for metric2 in enumerate(metrics[i+1:], i+1):
                if all(metric1 in r and metric2 in r for r in results):
                    trade_offs[f"{metric1}_vs_{metric2}"] = {
                        'correlation': stats.pearsonr(
                            [r[metric1] for r in results],
                            [r[metric2] for r in results]
                        )[0],
                        'pareto_optimal': self._find_pareto_optimal(
                            configs,
                            results,
                            [metric1, metric2]
                        )
                    }
                    
        return trade_offs
    
    def _find_pareto_optimal(self, configs: List[Dict],
                           results: List[Dict],
                           metrics: List[str]) -> List[int]:
        """Find Pareto optimal configurations for given metrics.
        
        Args:
            configs (List[Dict]): System configurations
            results (List[Dict]): Performance results
            metrics (List[str]): Metrics to consider
            
        Returns:
            List[int]: Indices of Pareto optimal configurations
        """
        n = len(configs)
        pareto_optimal = []
        
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j:
                    better_all = True
                    for metric in metrics:
                        if results[j][metric] <= results[i][metric]:
                            better_all = False
                            break
                    if better_all:
                        dominated = True
                        break
            if not dominated:
                pareto_optimal.append(i)
                
        return pareto_optimal

    def _test_significance(self, results: List[Dict]) -> Dict:
        """Test statistical significance of differences.
        
        Args:
            results (List[Dict]): Results to test
            
        Returns:
            Dict: Significance test results
        """
        significance = {}
        for metric in self.config.metrics_of_interest:
            if all(metric in r for r in results):
                # ANOVA test
                groups = [r[metric] for r in results]
                f_stat, p_value = stats.f_oneway(*groups)
                
                significance[metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level
                }
                
        return significance 