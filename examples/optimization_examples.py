"""
Examples demonstrating optimization capabilities of the hypercube model:
1. District boundary optimization
2. Unit location optimization
3. Dispatch policy optimization
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.zero_line_model import ZeroLineModel
from src.utils.config import ConfigManager, ModelConfig
from src.utils.logging_utils import ModelLogger
from src.visualization.figures import HypercubePlotter
from src.analysis.optimization import (
    DistrictOptimizer,
    LocationOptimizer,
    DispatchOptimizer
)

class OptimizationExample:
    """Demonstrates optimization capabilities of the hypercube model."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize example.
        
        Args:
            output_dir (Optional[str]): Output directory
        """
        self.output_dir = Path(output_dir or 'results/optimization')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = ModelLogger(name="OptimizationExamples")
        self.logger.setup_file_logging(self.output_dir / 'logs')
        self.plotter = HypercubePlotter(style='paper')
        
    def optimize_districts(self, model: ZeroLineModel, 
                         objective: str = 'workload_balance',
                         max_iterations: int = 100) -> Dict:
        """Optimize district boundaries.
        
        Args:
            model (ZeroLineModel): Initial model
            objective (str): Optimization objective
            max_iterations (int): Maximum iterations
            
        Returns:
            Dict: Optimization results
        """
        optimizer = DistrictOptimizer(
            model=model,
            objective=objective,
            max_iterations=max_iterations,
            logger=self.logger
        )
        
        with self.logger.time_operation('district_optimization'):
            # Run optimization
            results = optimizer.optimize()
            
            # Plot optimization progress
            self._plot_optimization_progress(
                results['progress'],
                title="District Optimization Progress",
                filename="district_optimization.pdf"
            )
            
            # Plot before/after comparison
            self._plot_district_comparison(
                results['initial'],
                results['optimized'],
                filename="district_comparison.pdf"
            )
            
        return results
    
    def optimize_locations(self, model: ZeroLineModel,
                         objective: str = 'response_time',
                         max_iterations: int = 100) -> Dict:
        """Optimize unit locations.
        
        Args:
            model (ZeroLineModel): Initial model
            objective (str): Optimization objective
            max_iterations (int): Maximum iterations
            
        Returns:
            Dict: Optimization results
        """
        optimizer = LocationOptimizer(
            model=model,
            objective=objective,
            max_iterations=max_iterations,
            logger=self.logger
        )
        
        with self.logger.time_operation('location_optimization'):
            # Run optimization
            results = optimizer.optimize()
            
            # Plot optimization progress
            self._plot_optimization_progress(
                results['progress'],
                title="Location Optimization Progress",
                filename="location_optimization.pdf"
            )
            
            # Plot location changes
            self._plot_location_changes(
                results['initial_locations'],
                results['optimized_locations'],
                filename="location_changes.pdf"
            )
            
        return results
    
    def optimize_dispatch_policy(self, model: ZeroLineModel,
                               objective: str = 'combined',
                               max_iterations: int = 100) -> Dict:
        """Optimize dispatch policy.
        
        Args:
            model (ZeroLineModel): Initial model
            objective (str): Optimization objective
            max_iterations (int): Maximum iterations
            
        Returns:
            Dict: Optimization results
        """
        optimizer = DispatchOptimizer(
            model=model,
            objective=objective,
            max_iterations=max_iterations,
            logger=self.logger
        )
        
        with self.logger.time_operation('dispatch_optimization'):
            # Run optimization
            results = optimizer.optimize()
            
            # Plot optimization progress
            self._plot_optimization_progress(
                results['progress'],
                title="Dispatch Policy Optimization Progress",
                filename="dispatch_optimization.pdf"
            )
            
            # Plot policy comparison
            self._plot_policy_comparison(
                results['initial_policy'],
                results['optimized_policy'],
                filename="policy_comparison.pdf"
            )
            
        return results
    
    def _plot_optimization_progress(self, progress: List[float],
                                 title: str, filename: str):
        """Plot optimization progress.
        
        Args:
            progress (List[float]): Objective values
            title (str): Plot title
            filename (str): Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.plot(progress, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(title)
        plt.grid(True)
        plt.savefig(self.output_dir / filename)
        plt.close()
        
    def _plot_district_comparison(self, initial: Dict, optimized: Dict,
                               filename: str):
        """Plot district configuration comparison.
        
        Args:
            initial (Dict): Initial configuration
            optimized (Dict): Optimized configuration
            filename (str): Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot initial configuration
        self.plotter.plot_district_map(
            initial['district_manager'],
            initial['atom_manager'],
            ax=ax1
        )
        ax1.set_title('Initial Districts')
        
        # Plot optimized configuration
        self.plotter.plot_district_map(
            optimized['district_manager'],
            optimized['atom_manager'],
            ax=ax2
        )
        ax2.set_title('Optimized Districts')
        
        plt.savefig(self.output_dir / filename)
        plt.close()
        
    def _plot_location_changes(self, initial: np.ndarray, optimized: np.ndarray,
                            filename: str):
        """Plot unit location changes.
        
        Args:
            initial (numpy.ndarray): Initial locations
            optimized (numpy.ndarray): Optimized locations
            filename (str): Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot initial locations
        ax.scatter(initial[:,0], initial[:,1], c='blue', label='Initial', s=100)
        
        # Plot optimized locations
        ax.scatter(optimized[:,0], optimized[:,1], c='red', label='Optimized', s=100)
        
        # Draw arrows showing movement
        for i in range(len(initial)):
            ax.arrow(initial[i,0], initial[i,1],
                    optimized[i,0] - initial[i,0],
                    optimized[i,1] - initial[i,1],
                    head_width=0.05, head_length=0.1, fc='k', ec='k')
            
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Unit Location Changes')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(self.output_dir / filename)
        plt.close()
        
    def _plot_policy_comparison(self, initial: Dict, optimized: Dict,
                             filename: str):
        """Plot dispatch policy comparison.
        
        Args:
            initial (Dict): Initial policy results
            optimized (Dict): Optimized policy results
            filename (str): Output filename
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Response time comparison
        ax1 = fig.add_subplot(221)
        ax1.bar(['Initial', 'Optimized'],
                [initial['mean_response_time'],
                 optimized['mean_response_time']])
        ax1.set_title('Mean Response Time')
        
        # Workload balance comparison
        ax2 = fig.add_subplot(222)
        ax2.bar(['Initial', 'Optimized'],
                [initial['workload_imbalance'],
                 optimized['workload_imbalance']])
        ax2.set_title('Workload Imbalance')
        
        # Interdistrict dispatch comparison
        ax3 = fig.add_subplot(223)
        ax3.bar(['Initial', 'Optimized'],
                [initial['interdistrict_fraction'],
                 optimized['interdistrict_fraction']])
        ax3.set_title('Interdistrict Dispatch Fraction')
        
        # Coverage comparison
        ax4 = fig.add_subplot(224)
        ax4.bar(['Initial', 'Optimized'],
                [initial['coverage'],
                 optimized['coverage']])
        ax4.set_title('Coverage')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
def main():
    """Run optimization examples."""
    example = OptimizationExample()
    
    try:
        # Create initial model
        config = ConfigManager().setup_default_config()
        model = ZeroLineModel(config)
        model.setup_linear_command()
        
        # Run district optimization
        district_results = example.optimize_districts(
            model,
            objective='workload_balance'
        )
        
        # Run location optimization
        location_results = example.optimize_locations(
            model,
            objective='response_time'
        )
        
        # Run dispatch policy optimization
        dispatch_results = example.optimize_dispatch_policy(
            model,
            objective='combined'
        )
        
        # Save results
        results = {
            'district_optimization': district_results,
            'location_optimization': location_results,
            'dispatch_optimization': dispatch_results,
            'timestamp': datetime.now().isoformat()
        }
        
        np.save(example.output_dir / 'optimization_results.npy', results)
        
        print("Optimization examples completed. Results saved in 'results/optimization'")
        
    except Exception as e:
        example.logger.log_error_with_context(e, {
            'phase': 'optimization_examples'
        })
        raise

if __name__ == "__main__":
    main()