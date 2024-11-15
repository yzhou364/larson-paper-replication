"""
Performance metrics analysis for hypercube queuing model.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from scipy import stats

@dataclass
class MetricsConfig:
    """Configuration for metrics analysis."""
    confidence_level: float = 0.95
    time_window: Optional[int] = None
    include_spatial: bool = True
    include_temporal: bool = True

class PerformanceAnalyzer:
    """Analyzes system performance metrics."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize performance analyzer.
        
        Args:
            config (Optional[MetricsConfig]): Analysis configuration
        """
        self.config = config or MetricsConfig()
        
    def analyze_workloads(self, workloads: np.ndarray) -> Dict:
        """Analyze unit workload distribution.
        
        Args:
            workloads (numpy.ndarray): Unit workloads
            
        Returns:
            Dict: Workload analysis results
        """
        analysis = {
            'basic_stats': {
                'mean': np.mean(workloads),
                'std': np.std(workloads),
                'min': np.min(workloads),
                'max': np.max(workloads),
                'median': np.median(workloads)
            },
            'imbalance': {
                'absolute': np.max(workloads) - np.min(workloads),
                'relative': (np.max(workloads) - np.min(workloads)) / np.mean(workloads),
                'cv': stats.variation(workloads)  # Coefficient of variation
            }
        }
        
        # Compute confidence intervals
        if len(workloads) > 1:
            ci = stats.t.interval(
                self.config.confidence_level,
                len(workloads)-1,
                loc=np.mean(workloads),
                scale=stats.sem(workloads)
            )
            analysis['confidence_interval'] = {
                'lower': ci[0],
                'upper': ci[1]
            }
            
        return analysis
    
    def analyze_response_times(self, times: np.ndarray, 
                             districts: Optional[np.ndarray] = None) -> Dict:
        """Analyze response time distribution.
        
        Args:
            times (numpy.ndarray): Response times
            districts (Optional[numpy.ndarray]): District assignments
            
        Returns:
            Dict: Response time analysis
        """
        analysis = {
            'basic_stats': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times),
                'percentile_90': np.percentile(times, 90)
            }
        }
        
        # Temporal analysis
        if self.config.include_temporal and len(times) > 1:
            analysis['temporal'] = self._analyze_temporal_pattern(times)
            
        # Spatial analysis by district
        if self.config.include_spatial and districts is not None:
            analysis['spatial'] = self._analyze_spatial_pattern(times, districts)
            
        return analysis
    
    def analyze_queue_performance(self, queue_data: Dict) -> Dict:
        """Analyze queue performance metrics.
        
        Args:
            queue_data (Dict): Queue performance data
            
        Returns:
            Dict: Queue analysis results
        """
        analysis = {
            'queue_length': {
                'mean': np.mean(queue_data['lengths']),
                'max': np.max(queue_data['lengths']),
                'zero_probability': np.mean(queue_data['lengths'] == 0)
            },
            'wait_times': {
                'mean': np.mean(queue_data['wait_times']),
                'median': np.median(queue_data['wait_times']),
                'percentile_90': np.percentile(queue_data['wait_times'], 90)
            }
        }
        
        # Little's Law verification
        arrival_rate = queue_data.get('arrival_rate')
        if arrival_rate:
            expected_queue = arrival_rate * np.mean(queue_data['wait_times'])
            analysis['littles_law'] = {
                'observed': np.mean(queue_data['lengths']),
                'expected': expected_queue,
                'relative_error': abs(expected_queue - np.mean(queue_data['lengths'])) / expected_queue
            }
            
        return analysis
    
    def analyze_coverage(self, times: np.ndarray, 
                        threshold: float,
                        weights: Optional[np.ndarray] = None) -> Dict:
        """Analyze coverage metrics.
        
        Args:
            times (numpy.ndarray): Response times
            threshold (float): Coverage time threshold
            weights (Optional[numpy.ndarray]): Demand weights
            
        Returns:
            Dict: Coverage analysis results
        """
        # Basic coverage calculation
        coverage_mask = times <= threshold
        
        if weights is None:
            weights = np.ones_like(times) / len(times)
            
        analysis = {
            'coverage_fraction': np.sum(coverage_mask * weights),
            'mean_excess': np.mean(np.maximum(times - threshold, 0)),
            'max_excess': np.max(np.maximum(times - threshold, 0))
        }
        
        # Reliability analysis
        if self.config.time_window:
            reliability = self._analyze_reliability(
                times, threshold, self.config.time_window
            )
            analysis['reliability'] = reliability
            
        return analysis
    
    def _analyze_temporal_pattern(self, data: np.ndarray) -> Dict:
        """Analyze temporal patterns in data.
        
        Args:
            data (numpy.ndarray): Time series data
            
        Returns:
            Dict: Temporal analysis results
        """
        df = pd.Series(data)
        
        return {
            'trend': {
                'slope': stats.linregress(np.arange(len(data)), data).slope,
                'rolling_mean': df.rolling(window=min(len(data)//10, 100)).mean().values
            },
            'seasonality': {
                'hourly': self._detect_seasonality(data, period=24),
                'daily': self._detect_seasonality(data, period=24*7)
            },
            'stationarity': self._test_stationarity(data)
        }
        
    def _analyze_spatial_pattern(self, data: np.ndarray, 
                               districts: np.ndarray) -> Dict:
        """Analyze spatial patterns in data.
        
        Args:
            data (numpy.ndarray): Data values
            districts (numpy.ndarray): District assignments
            
        Returns:
            Dict: Spatial analysis results
        """
        unique_districts = np.unique(districts)
        district_means = [np.mean(data[districts == d]) for d in unique_districts]
        
        return {
            'district_stats': {
                'means': district_means,
                'variation': stats.variation(district_means)
            },
            'spatial_autocorrelation': self._compute_spatial_autocorrelation(
                data, districts
            )
        }
        
    def _detect_seasonality(self, data: np.ndarray, period: int) -> Dict:
        """Detect seasonality in time series.
        
        Args:
            data (numpy.ndarray): Time series data
            period (int): Seasonality period to test
            
        Returns:
            Dict: Seasonality analysis results
        """
        if len(data) < 2 * period:
            return {'detected': False}
            
        # Compute autocorrelation
        acf = pd.Series(data).autocorr(lag=period)
        
        return {
            'detected': abs(acf) > 0.2,
            'strength': abs(acf),
            'period': period
        }
        
    def _test_stationarity(self, data: np.ndarray) -> Dict:
        """Test for stationarity in time series.
        
        Args:
            data (numpy.ndarray): Time series data
            
        Returns:
            Dict: Stationarity test results
        """
        # Augmented Dickey-Fuller test
        adf_result = stats.adfuller(data)
        
        return {
            'is_stationary': adf_result[1] < 0.05,
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4]
        }
        
    def _compute_spatial_autocorrelation(self, data: np.ndarray,
                                       districts: np.ndarray) -> float:
        """Compute spatial autocorrelation.
        
        Args:
            data (numpy.ndarray): Data values
            districts (numpy.ndarray): District assignments
            
        Returns:
            float: Spatial autocorrelation coefficient
        """
        # Simple implementation of Moran's I
        n = len(np.unique(districts))
        mean = np.mean(data)
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    i_data = data[districts == i]
                    j_data = data[districts == j]
                    
                    if len(i_data) > 0 and len(j_data) > 0:
                        numerator += (np.mean(i_data) - mean) * (np.mean(j_data) - mean)
                        
            denominator += (np.mean(data[districts == i]) - mean) ** 2
            
        if denominator == 0:
            return 0
            
        return numerator / (denominator * n)
    
    def _analyze_reliability(self, times: np.ndarray,
                           threshold: float,
                           window: int) -> Dict:
        """Analyze system reliability over time windows.
        
        Args:
            times (numpy.ndarray): Response times
            threshold (float): Time threshold
            window (int): Analysis window size
            
        Returns:
            Dict: Reliability analysis results
        """
        # Split data into windows
        num_windows = len(times) // window
        reliabilities = []
        
        for i in range(num_windows):
            window_data = times[i*window:(i+1)*window]
            reliability = np.mean(window_data <= threshold)
            reliabilities.append(reliability)
            
        return {
            'mean': np.mean(reliabilities),
            'min': np.min(reliabilities),
            'std': np.std(reliabilities),
            'below_90': np.mean(np.array(reliabilities) < 0.9)
        }