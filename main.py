"""
Main entry point for Larson's Hypercube Queuing Model implementation.
Replicates the analysis from the 1974 paper and provides additional features.
"""

import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse
from typing import Dict, Optional

from src.models.zero_line_model import ZeroLineModel
from src.models.infinite_line_model import InfiniteLineModel
from src.utils.config import ConfigManager
from src.utils.logging_utils import ModelLogger
from src.visualization.figures import HypercubePlotter
from src.visualization.interactive import InteractiveVisualizer

class HypercubeAnalysis:
    """Main analysis class for hypercube model."""
    
    def __init__(self, output_dir: str = 'results'):
        """Initialize analysis.
        
        Args:
            output_dir (str): Output directory, defaults to 'results'
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = ModelLogger(name="HypercubeAnalysis")
        self.logger.setup_file_logging(self.output_dir / 'logs')
        
        # Setup visualization
        self.plotter = HypercubePlotter(style='paper')
        self.interactive_viz = InteractiveVisualizer()
        
        # Load configuration with default values
        config_manager = ConfigManager()
        self.config = config_manager.setup_default_config()
        
        # Initialize results storage
        self.results = {
            'zero_line': {},
            'infinite_line': {},
            'comparisons': {}
        }
        
    def run_paper_replication(self):
        """Replicate analysis from Larson's 1974 paper."""
        self.logger.log_event('analysis_start', {'type': 'paper_replication'})
        
        try:
            # Run analysis for different utilization levels
            rho_values = np.linspace(0.1, 0.9, 9)
            
            for rho in rho_values:
                self.logger.log_event('iteration_start', {'rho': rho})
                
                # Adjust arrival rate for desired utilization
                lambda_rate = rho * self.config.system.N
                self.config.system.lambda_rate = lambda_rate
                
                # Run zero-line capacity model
                zero_line = ZeroLineModel(self.config)
                zero_line.setup_linear_command()
                zero_results = zero_line.run()
                
                # Run infinite-line capacity model
                infinite_line = InfiniteLineModel(self.config)
                infinite_line.setup_linear_command()
                infinite_results = infinite_line.run()
                
                # Store results
                self.results['zero_line'][rho] = zero_results
                self.results['infinite_line'][rho] = infinite_results
                
                self.logger.log_event('iteration_complete', {'rho': rho})
                
            # Generate comparisons
            self._generate_comparisons()
            
            # Create visualizations
            self._generate_visualizations()
            
            # Generate report
            self._generate_report()
            
        except Exception as e:
            self.logger.log_error_with_context(e, {'phase': 'paper_replication'})
            raise
            
    def run_custom_analysis(self, analysis_type: str, **kwargs):
        """Run custom analysis configuration.
        
        Args:
            analysis_type (str): Type of analysis to run
            **kwargs: Additional analysis parameters
        """
        self.logger.log_event('analysis_start', {
            'type': 'custom',
            'analysis_type': analysis_type
        })
        
        try:
            if analysis_type == 'grid':
                self._run_grid_analysis(**kwargs)
            elif analysis_type == 'optimization':
                self._run_optimization_analysis(**kwargs)
            elif analysis_type == 'sensitivity':
                self._run_sensitivity_analysis(**kwargs)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            self.logger.log_error_with_context(e, {
                'phase': 'custom_analysis',
                'type': analysis_type
            })
            raise
            
    def _generate_comparisons(self):
        """Generate system comparisons."""
        from src.analysis.comparison import SystemComparison
        
        comparator = SystemComparison()
        self.results['comparisons'] = comparator.compare_queue_types(
            self.results['zero_line'],
            self.results['infinite_line']
        )
        
    def _generate_visualizations(self):
        """Generate paper figures and interactive visualizations."""
        # Extract results
        rho_values = sorted(self.results['zero_line'].keys())
        
        # Create paper figures
        self.plotter.plot_workload_distribution(
            [r['workloads'] for r in self.results['zero_line'].values()],
            title="Workload Distribution (Zero-line)"
        )
        
        self.plotter.plot_workload_distribution(
            [r['workloads'] for r in self.results['infinite_line'].values()],
            title="Workload Distribution (Infinite-line)"
        )
        
        # Create interactive dashboard
        dashboard = self.interactive_viz.create_performance_dashboard(
            self.results, rho_values
        )
        
        # Save visualizations
        self.plotter.save_figures(
            self.output_dir / 'figures',
            formats=['pdf', 'png']
        )
        
        self.interactive_viz.save_figure(
            dashboard,
            self.output_dir / 'dashboard'
        )
        
    def _generate_report(self):
        """Generate analysis report."""
        report_path = self.output_dir / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Hypercube Model Analysis Report\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # System configuration
            f.write("## System Configuration\n")
            f.write(f"- Number of units: {self.config.system.N}\n")
            f.write(f"- Number of atoms: {self.config.system.J}\n")
            f.write("- Model types: Zero-line and Infinite-line capacity\n\n")
            
            # Performance results
            f.write("## Performance Results\n\n")
            for rho in sorted(self.results['zero_line'].keys()):
                f.write(f"\n### System Utilization ρ = {rho:.1f}\n")
                
                # Zero-line results
                zero_res = self.results['zero_line'][rho]
                f.write("\nZero-line Capacity Model:\n")
                f.write(f"- Mean workload: {np.mean(zero_res['workloads']):.3f}\n")
                f.write(f"- Workload imbalance: {zero_res['workload_imbalance']:.3f}\n")
                
                # Infinite-line results
                inf_res = self.results['infinite_line'][rho]
                f.write("\nInfinite-line Capacity Model:\n")
                f.write(f"- Mean workload: {np.mean(inf_res['workloads']):.3f}\n")
                f.write(f"- Workload imbalance: {inf_res['workload_imbalance']:.3f}\n")
                if inf_res.get('queue_metrics'):
                    f.write(f"- Mean queue length: {inf_res['queue_metrics']['mean_queue_length']:.3f}\n")
                    
            # Comparisons
            f.write("\n## System Comparisons\n\n")
            comparisons = self.results['comparisons']
            for metric, comparison in comparisons.items():
                f.write(f"\n### {metric}\n")
                f.write(f"- Relative difference: {comparison['relative_change']:.2%}\n")
                if comparison.get('statistical_tests'):
                    f.write(f"- Statistical significance: p = {comparison['statistical_tests']['p_value']:.4f}\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hypercube Queuing Model Analysis"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['paper', 'grid', 'optimization', 'sensitivity'],
        default='paper',
        help='Type of analysis to run'
    )
    
    args = parser.parse_args()
    
    # Create analysis instance with specified output directory
    analysis = HypercubeAnalysis(output_dir=args.output)
    
    try:
        if args.analysis == 'paper':
            analysis.run_paper_replication()
        else:
            analysis.run_custom_analysis(args.analysis)
            
        print(f"Analysis completed successfully! Results saved in: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()