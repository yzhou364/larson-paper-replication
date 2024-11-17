"""
Main entry point for Larson's Hypercube Queuing Model implementation.
Replicates the analysis from the 1974 paper and provides additional features.
"""

import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional
from src.utils.config_adapter import adapt_config
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
                
                # Create adapted config for models
                model_config = adapt_config(self.config)
                
                # Run zero-line capacity model
                zero_line = ZeroLineModel(model_config)
                zero_line.setup_linear_command()
                zero_results = zero_line.run()
                
                # Store results with proper structure
                self.results['zero_line'][rho] = {
                    'workloads': zero_results.workloads,
                    'travel_times': zero_results.travel_times,
                    'interdistrict_fraction': zero_results.interdistrict_fractions
                }
                
                # Run infinite-line capacity model
                infinite_line = InfiniteLineModel(model_config)
                infinite_line.setup_linear_command()
                infinite_results = infinite_line.run()
                
                # Store results with proper structure
                self.results['infinite_line'][rho] = {
                    'workloads': infinite_results.workloads,
                    'travel_times': infinite_results.travel_times,
                    'interdistrict_fraction': infinite_results.interdistrict_fractions,
                    'queue_metrics': infinite_results.queue_metrics
                }
                
                self.logger.log_event('iteration_complete', {'rho': rho})
                
            # Generate comparisons
            self._generate_comparisons()
            
            # Generate visualizations
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
        from src.analysis.comparison import SystemComparator
        
        comparator = SystemComparator()
        self.results['comparisons'] = comparator.compare_queue_types(
            self.results['zero_line'],
            self.results['infinite_line']
        )
        
    def _generate_visualizations(self):
        """Generate paper figures and interactive visualizations."""
        # Extract results for all rho values
        rho_values = sorted(self.results['zero_line'].keys())
        latest_rho = rho_values[-1]  # Get most recent rho value
        
        # Create figure directory
        figure_dir = self.output_dir / 'figures'
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot workload distributions
        self.plotter.plot_workload_distribution(
            self.results['zero_line'][latest_rho]['workloads'],
            title="Workload Distribution (Zero-line)"
        )
        self.plotter.save_figures(figure_dir / 'zero_line_workload', formats=['pdf', 'png'])
        
        self.plotter.plot_workload_distribution(
            self.results['infinite_line'][latest_rho]['workloads'],
            title="Workload Distribution (Infinite-line)"
        )
        self.plotter.save_figures(figure_dir / 'infinite_line_workload', formats=['pdf', 'png'])
        
        # Plot performance comparison
        self.plotter.plot_performance_comparison(
            self.results['zero_line'],
            self.results['infinite_line'],
            rho_values
        )
        self.plotter.save_figures(figure_dir / 'performance_comparison', formats=['pdf', 'png'])
        
        # Create interactive dashboard
        dashboard = self.interactive_viz.create_performance_dashboard(self.results)
        self.interactive_viz.save_figure(dashboard, self.output_dir / 'dashboard')
        
        plt.close('all')  

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
            
            # Get all rho values and the latest one for current state
            rho_values = sorted(self.results['zero_line'].keys())
            latest_rho = rho_values[-1]
            
            for rho in sorted(self.results['zero_line'].keys()):
                f.write(f"\n### System Utilization ρ = {rho:.1f}\n")
                
                # Zero-line results
                zero_res = self.results['zero_line'][rho]
                f.write("\nZero-line Capacity Model:\n")
                f.write(f"- Mean workload: {np.mean(zero_res['workloads']):.3f}\n")
                f.write(f"- Workload imbalance: {(np.max(zero_res['workloads']) - np.min(zero_res['workloads'])) / np.mean(zero_res['workloads']):.3f}\n")
                f.write(f"- Mean travel time: {zero_res['travel_times']['average']:.3f}\n")
                f.write(f"- Mean interdistrict fraction: {np.mean(zero_res['interdistrict_fraction']):.3f}\n")
                
                # Infinite-line results
                inf_res = self.results['infinite_line'][rho]
                f.write("\nInfinite-line Capacity Model:\n")
                f.write(f"- Mean workload: {np.mean(inf_res['workloads']):.3f}\n")
                f.write(f"- Workload imbalance: {(np.max(inf_res['workloads']) - np.min(inf_res['workloads'])) / np.mean(inf_res['workloads']):.3f}\n")
                f.write(f"- Mean travel time: {inf_res['travel_times']['average']:.3f}\n")
                f.write(f"- Mean interdistrict fraction: {np.mean(inf_res['interdistrict_fraction']):.3f}\n")
                
                # Queue metrics for infinite-line model
                if 'queue_metrics' in inf_res:
                    queue_metrics = inf_res['queue_metrics']
                    f.write("\nQueue Performance:\n")
                    f.write(f"- Expected queue length: {queue_metrics['expected_queue_length']:.3f}\n")
                    f.write(f"- Expected wait time: {queue_metrics['expected_wait_time']:.3f}\n")
                    f.write(f"- Queue probability: {queue_metrics['probability_queue']:.3f}\n")
                        
            # System comparisons section
            f.write("\n## System Comparisons\n\n")
            if 'comparisons' in self.results:
                comparisons = self.results['comparisons']
                
                if 'workload' in comparisons:
                    f.write("### Workload Comparison\n")
                    avg_change = comparisons['workload']['average']['relative_change']
                    f.write(f"- Average relative change: {avg_change:.2%}\n")
                    
                if 'response_time' in comparisons:
                    f.write("\n### Response Time Comparison\n")
                    avg_change = comparisons['response_time']['average']['relative_change']
                    f.write(f"- Average relative change: {avg_change:.2%}\n")
                    
                if 'interdistrict' in comparisons:
                    f.write("\n### Interdistrict Response Comparison\n")
                    avg_change = comparisons['interdistrict']['average']['relative_change']
                    f.write(f"- Average relative change: {avg_change:.2%}\n")
            
            # Summary and recommendations
            f.write("\n## Summary and Recommendations\n\n")
            
            # Calculate overall performance differences
            latest_zero = self.results['zero_line'][latest_rho]
            latest_inf = self.results['infinite_line'][latest_rho]
            
            workload_diff = (np.mean(latest_inf['workloads']) - np.mean(latest_zero['workloads'])) / np.mean(latest_zero['workloads'])
            travel_diff = (latest_inf['travel_times']['average'] - latest_zero['travel_times']['average']) / latest_zero['travel_times']['average']
            
            f.write("### Key Findings:\n")
            f.write(f"1. Workload Difference: {workload_diff:.2%}\n")
            f.write(f"2. Travel Time Difference: {travel_diff:.2%}\n")
            
            f.write("\n### Recommendations:\n")
            # Add recommendations based on results
            if abs(workload_diff) > 0.1:
                f.write("- Consider workload balancing strategies\n")
            if abs(travel_diff) > 0.1:
                f.write("- Review travel time optimization opportunities\n")
            if 'queue_metrics' in latest_inf and latest_inf['queue_metrics']['probability_queue'] > 0.2:
                f.write("- Monitor queue formation closely\n")

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