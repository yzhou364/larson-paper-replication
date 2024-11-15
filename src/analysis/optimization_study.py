"""
Comprehensive optimization study for hypercube model configurations.
Includes studies of:
1. District boundary optimization
2. Unit location optimization
3. Dispatch policy optimization
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.models.zero_line_model import ZeroLineModel
from src.utils.config import ConfigManager
from src.utils.logging_utils import ModelLogger
from src.visualization.figures import HypercubePlotter
from src.analysis.optimization import (
    DistrictOptimizer,
    LocationOptimizer,
    DispatchOptimizer,
    OptimizationConfig
)

class OptimizationStudy:
    """Conducts comprehensive optimization studies."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize optimization study.
        
        Args:
            output_dir (Optional[str]): Output directory
        """
        self.output_dir = Path(output_dir or 'results/optimization_study')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = ModelLogger(name="OptimizationStudy")
        self.logger.setup_file_logging(self.output_dir / 'logs')
        
        # Setup visualization
        self.plotter = HypercubePlotter(style='paper')
        
        # Load configuration
        self.config = ConfigManager().setup_default_config()
        
        # Initialize results storage
        self.results = {
            'district': {},
            'location': {},
            'dispatch': {}
        }
        
    def run_district_optimization(self):
        """Run district boundary optimization study."""
        self.logger.log_event('study_start', {'type': 'district_optimization'})
        
        try:
            # Initialize model
            model = ZeroLineModel(self.config)
            model.setup_linear_command()
            
            # Create optimizer
            optimizer = DistrictOptimizer(
                model=model,
                config=OptimizationConfig(
                    max_iterations=100,
                    tolerance=1e-6
                )
            )
            
            # Run optimization
            with self.logger.time_operation('district_optimization'):
                results = optimizer.optimize()
                
            # Save results
            self.results['district'] = results
            
            # Generate visualizations
            self._plot_district_optimization_results(results)
            
        except Exception as e:
            self.logger.log_error_with_context(e, {
                'phase': 'district_optimization'
            })
            raise
            
    def run_location_optimization(self):
        """Run unit location optimization study."""
        self.logger.log_event('study_start', {'type': 'location_optimization'})
        
        try:
            # Initialize model
            model = ZeroLineModel(self.config)
            model.setup_linear_command()
            
            # Create optimizer
            optimizer = LocationOptimizer(
                model=model,
                config=OptimizationConfig(
                    max_iterations=100,
                    population_size=50,
                    mutation_rate=0.2
                )
            )
            
            # Run optimization
            with self.logger.time_operation('location_optimization'):
                results = optimizer.optimize()
                
            # Save results
            self.results['location'] = results
            
            # Generate visualizations
            self._plot_location_optimization_results(results)
            
        except Exception as e:
            self.logger.log_error_with_context(e, {
                'phase': 'location_optimization'
            })
            raise
            
    def run_dispatch_optimization(self):
        """Run dispatch policy optimization study."""
        self.logger.log_event('study_start', {'type': 'dispatch_optimization'})
        
        try:
            # Initialize model
            model = ZeroLineModel(self.config)
            model.setup_linear_command()
            
            # Create optimizer
            optimizer = DispatchOptimizer(
                model=model,
                config=OptimizationConfig(
                    max_iterations=100,
                    multi_objective=True,
                    use_constraints=True
                )
            )
            
            # Run optimization
            with self.logger.time_operation('dispatch_optimization'):
                results = optimizer.optimize()
                
            # Save results
            self.results['dispatch'] = results
            
            # Generate visualizations
            self._plot_dispatch_optimization_results(results)
            
        except Exception as e:
            self.logger.log_error_with_context(e, {
                'phase': 'dispatch_optimization'
            })
            raise
            
    def _plot_district_optimization_results(self, results: Dict):
        """Plot district optimization results."""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot district configurations
        ax1 = fig.add_subplot(221)
        self.plotter.plot_district_map(
            results['initial_districts'],
            title="Initial Districts"
        )
        
        ax2 = fig.add_subplot(222)
        self.plotter.plot_district_map(
            results['optimized_districts'],
            title="Optimized Districts"
        )
        
        # Plot optimization progress
        ax3 = fig.add_subplot(223)
        progress = [h['objective'] for h in results['history']]
        ax3.plot(progress, 'b-')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective Value')
        ax3.set_title('Optimization Progress')
        
        # Plot performance comparison
        ax4 = fig.add_subplot(224)
        metrics = ['workload_imbalance', 'mean_travel_time', 'interdistrict_fraction']
        initial = [results['initial_metrics'][m] for m in metrics]
        optimized = [results['optimized_metrics'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax4.bar(x - width/2, initial, width, label='Initial')
        ax4.bar(x + width/2, optimized, width, label='Optimized')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45)
        ax4.legend()
        ax4.set_title('Performance Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'district_optimization.pdf')
        plt.close()
        
    def _plot_location_optimization_results(self, results: Dict):
        """Plot location optimization results."""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot unit locations
        ax1 = fig.add_subplot(221)
        self._plot_locations(
            results['initial_locations'],
            title="Initial Locations"
        )
        
        ax2 = fig.add_subplot(222)
        self._plot_locations(
            results['optimized_locations'],
            title="Optimized Locations"
        )
        
        # Plot optimization progress
        ax3 = fig.add_subplot(223)
        progress = [h['objective'] for h in results['history']]
        ax3.plot(progress, 'b-')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective Value')
        ax3.set_title('Optimization Progress')
        
        # Plot coverage comparison
        ax4 = fig.add_subplot(224)
        coverage_initial = self._compute_coverage(results['initial_locations'])
        coverage_optimized = self._compute_coverage(results['optimized_locations'])
        
        ax4.plot(coverage_initial['radii'], coverage_initial['coverage'],
                 'b-', label='Initial')
        ax4.plot(coverage_optimized['radii'], coverage_optimized['coverage'],
                 'r-', label='Optimized')
        ax4.set_xlabel('Coverage Radius')
        ax4.set_ylabel('Coverage Fraction')
        ax4.legend()
        ax4.set_title('Coverage Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'location_optimization.pdf')
        plt.close()
        
    def _plot_dispatch_optimization_results(self, results: Dict):
        """Plot dispatch optimization results."""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot optimization progress
        ax1 = fig.add_subplot(221)
        progress = [h['objective'] for h in results['history']]
        ax1.plot(progress, 'b-')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization Progress')
        
        # Plot Pareto front
        if results.get('pareto_front'):
            ax2 = fig.add_subplot(222)
            objectives = np.array([
                self._evaluate_policy_multi(p) for p in results['pareto_front']
            ])
            ax2.scatter(objectives[:,0], objectives[:,1], c=objectives[:,2],
                       cmap='viridis')
            ax2.set_xlabel('Travel Time')
            ax2.set_ylabel('Workload Balance')
            ax2.set_title('Pareto Front')
            plt.colorbar(ax2.collections[0], label='Interdistrict Fraction')
            
        # Plot policy parameter changes
        ax3 = fig.add_subplot(223)
        param_changes = results['parameter_changes']
        sns.heatmap(
            pd.DataFrame(param_changes).T,
            cmap='RdYlBu',
            center=0,
            ax=ax3
        )
        ax3.set_title('Parameter Changes')
        
        # Plot performance metrics
        ax4 = fig.add_subplot(224)
        metrics = ['travel_time', 'workload_balance', 'interdistrict']
        initial = [results['initial_metrics'][m] for m in metrics]
        optimized = [results['optimized_metrics'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax4.bar(x - width/2, initial, width, label='Initial')
        ax4.bar(x + width/2, optimized, width, label='Optimized')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45)
        ax4.legend()
        ax4.set_title('Performance Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dispatch_optimization.pdf')
        plt.close()
        
    def _plot_locations(self, locations: np.ndarray, title: str):
        """Plot unit locations."""
        plt.scatter(locations[:,0], locations[:,1], c='blue')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        
    def _compute_coverage(self, locations: np.ndarray) -> Dict:
        """Compute coverage at different radii."""
        radii = np.linspace(0, 1, 50)
        coverage = []
        
        for r in radii:
            covered = set()
            for loc in locations:
                for j in range(self.config.J):
                    atom = self.model.atom_manager.atoms[j]
                    if np.sqrt(((loc - atom.center)**2).sum()) <= r:
                        covered.add(j)
            coverage.append(len(covered) / self.config.J)
            
        return {
            'radii': radii,
            'coverage': coverage
        }
        
    def generate_report(self):
        """Generate optimization study report."""
        report_path = self.output_dir / 'optimization_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Hypercube Model Optimization Study\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # District optimization results
            f.write("## District Boundary Optimization\n\n")
            if self.results['district']:
                self._write_district_results(f)
                
            # Location optimization results
            f.write("\n## Unit Location Optimization\n\n")
            if self.results['location']:
                self._write_location_results(f)
                
            # Dispatch optimization results
            f.write("\n## Dispatch Policy Optimization\n\n")
            if self.results['dispatch']:
                self._write_dispatch_results(f)
                
            # Recommendations
            f.write("\n## Recommendations\n\n")
            self._write_recommendations(f)
            
    def _write_district_results(self, f):
        """Write district optimization results."""
        results = self.results['district']
        summary = results['summary']
        
        f.write(f"- Number of iterations: {summary['iterations']}\n")
        f.write(f"- Improvement: {summary['improvement']:.2%}\n")
        f.write(f"- Convergence achieved: {summary['convergence']}\n\n")
        
        f.write("### Performance Comparison\n\n")
        f.write("| Metric | Initial | Optimized | Improvement |\n")
        f.write("|--------|----------|------------|-------------|\n")
        
        metrics = ['workload_imbalance', 'mean_travel_time', 'interdistrict_fraction']
        for metric in metrics:
            initial = results['initial_metrics'][metric]
            optimized = results['optimized_metrics'][metric]
            improvement = (optimized - initial) / initial
            f.write(f"| {metric} | {initial:.3f} | {optimized:.3f} | {improvement:.2%} |\n")
            
    def _write_location_results(self, f):
        """Write location optimization results."""
        results = self.results['location']
        summary = results['summary']
        
        f.write(f"- Number of iterations: {summary['iterations']}\n")
        f.write(f"- Improvement: {summary['improvement']:.2%}\n")
        f.write(f"- Convergence achieved: {summary['convergence']}\n\n")
        
        f.write("### Coverage Improvement\n\n")
        initial_coverage = self._compute_coverage(results['initial_locations'])
        optimized_coverage = self._compute_coverage(results['optimized_locations'])
        
        f.write(f"- Initial coverage (r=0.5): {initial_coverage['coverage'][25]:.2%}\n")
        f.write(f"- Optimized coverage (r=0.5): {optimized_coverage['coverage'][25]:.2%}\n")
        f.write(f"- Coverage improvement: {(optimized_coverage['coverage'][25] - initial_coverage['coverage'][25]) / initial_coverage['coverage'][25]:.2%}\n")
        
    def _write_dispatch_results(self, f):
        """Write dispatch optimization results."""
        results = self.results['dispatch']
        
        if results.get('pareto_front'):
            f.write("### Pareto-Optimal Solutions\n\n")
            f.write("| Travel Time | Workload Balance | Interdistrict Fraction |\n")
            f.write("|-------------|------------------|----------------------|\n")
            
            for policy in results['pareto_front'][:5]:  # Show top 5
                objectives = self._evaluate_policy_multi(policy)
                f.write(f"| {objectives[0]:.3f} | {objectives[1]:.3f} | {objectives[2]:.3f} |\n")
                
        f.write("\n### Policy Parameter Changes\n\n")
        f.write("| Parameter | Initial | Optimized | Change |\n")
        f.write("|-----------|----------|-----------|--------|\n")
        
        for param, change in results['parameter_changes'].items():
            f.write(f"| {param} | {change['initial']:.3f} | {change['final']:.3f} | {change['change']:.2%} |\n")
            
    def _write_recommendations(self, f):
        """Write optimization recommendations."""
        f.write("### District Recommendations\n\n")
        if self.results['district']:
            district_rec = self.results['district'].get('recommendations', {})
            for rec in district_rec.get('implementation_steps', []):
                f.write(f"- {rec['action']}: {rec.get('details', '')}\n")
                
        f.write("\n### Location Recommendations\n\n")
        if self.results['location']:
            location_rec = self.results['location'].get('recommendations', {})
            for rec in location_rec.get('implementation_steps', []):
                f.write(f"- {rec['action']}: {rec.get('details', '')}\n")
                
        f.write("\n### Dispatch Policy Recommendations\n\n")
        if self.results['dispatch']:
            policy_rec = self.results['dispatch'].get('recommendations', {})
            for rec in policy_rec.get('implementation_steps', []):
                f.write(f"- {rec['action']}: {rec.get('details', '')}\n")
                
    def save_results(self):
        """Save optimization results to file."""
        # Save numerical results
        np.save(
            self.output_dir / 'optimization_results.npy',
            self.results
        )
        
        # Save optimization history
        history = {
            'district': [h for h in self.results.get('district', {}).get('history', [])],
            'location': [h for h in self.results.get('location', {}).get('history', [])],
            'dispatch': [h for h in self.results.get('dispatch', {}).get('history', [])]
        }
        np.save(
            self.output_dir / 'optimization_history.npy',
            history
        )
        
        # Save performance metrics
        metrics = {
            'district': self._extract_metrics(self.results.get('district', {})),
            'location': self._extract_metrics(self.results.get('location', {})),
            'dispatch': self._extract_metrics(self.results.get('dispatch', {}))
        }
        np.save(
            self.output_dir / 'performance_metrics.npy',
            metrics
        )
        
    def _extract_metrics(self, results: Dict) -> Dict:
        """Extract performance metrics from results."""
        if not results:
            return {}
            
        return {
            'initial': results.get('initial_metrics', {}),
            'optimized': results.get('optimized_metrics', {}),
            'improvement': results.get('summary', {}).get('improvement', 0)
        }
        
def main():
    """Run optimization study."""
    # Create output directory
    output_dir = Path('results/optimization_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize study
    study = OptimizationStudy(output_dir=output_dir)
    
    try:
        # Run district optimization
        study.run_district_optimization()
        
        # Run location optimization
        study.run_location_optimization()
        
        # Run dispatch optimization
        study.run_dispatch_optimization()
        
        # Generate report and save results
        study.generate_report()
        study.save_results()
        
        print("Optimization study completed successfully!")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        study.logger.log_error_with_context(e, {
            'phase': 'optimization_study'
        })
        raise

if __name__ == "__main__":
    main()