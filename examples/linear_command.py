"""
Implementation of the linear command example from Larson's 1974 paper.
This example demonstrates a nine-district linear command with 18 atoms.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Optional

from src.models.zero_line_model import ZeroLineModel
from src.models.infinite_line_model import InfiniteLineModel
from src.utils.config import ConfigManager, ModelConfig, SystemConfig, GeometryConfig
from src.utils.logging_utils import ModelLogger
from src.visualization.figures import HypercubePlotter
from src.visualization.interactive import InteractiveVisualizer

class LinearCommandExample:
    """Implements the linear command example from the paper."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize example.
        
        Args:
            output_dir (Optional[str]): Directory for outputs
        """
        self.output_dir = Path(output_dir or 'results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = ModelLogger(
            name="LinearCommand",
            config=None  # Use default config
        )
        self.logger.setup_file_logging(self.output_dir / 'logs')
        
        # Setup visualization
        self.plotter = HypercubePlotter(style='paper')
        self.interactive_viz = InteractiveVisualizer()
        
        # Initialize results storage
        self.results = {
            'zero_line': {},
            'infinite_line': {}
        }
        
    def setup_config(self) -> ModelConfig:
        """Setup model configuration.
        
        Returns:
            ModelConfig: Model configuration
        """
        config_manager = ConfigManager()
        
        # Create basic configuration
        system_config = SystemConfig(
            N=9,  # 9 districts
            J=18,  # 18 atoms
            lambda_rate=1.0,  # Will be adjusted for different ρ values
            mu_rate=1.0
        )
        
        geometry_config = GeometryConfig(
            district_length=1.0,
            is_grid=False  # Linear configuration
        )
        
        # Load and customize configuration
        config_path = Path(__file__).parent / 'configs' / 'linear_command.yaml'
        if config_path.exists():
            config_manager.load_config(config_path)
        else:
            # Create default configuration
            config_manager.config = ModelConfig(
                system=system_config,
                geometry=geometry_config,
                computation=None,  # Use defaults
                output=None       # Use defaults
            )
            
        return config_manager.get_config()
        
    def run_analysis(self, rho_values: Optional[np.ndarray] = None):
        """Run analysis for different system utilization values.
        
        Args:
            rho_values (Optional[numpy.ndarray]): System utilization values
        """
        if rho_values is None:
            rho_values = np.linspace(0.1, 0.9, 9)
            
        config = self.setup_config()
        N = config.system.N
        
        for rho in rho_values:
            self.logger.log_event('analysis_start', {'rho': rho})
            
            # Adjust arrival rate for desired utilization
            config.system.lambda_rate = rho * N  # Since μ = 1.0
            
            with self.logger.time_operation(f'analysis_rho_{rho}'):
                # Run zero-line capacity model
                zero_line = ZeroLineModel(config)
                zero_line.setup_linear_command()
                zero_results = zero_line.run()
                
                # Run infinite-line capacity model
                infinite_line = InfiniteLineModel(config)
                infinite_line.setup_linear_command()
                infinite_results = infinite_line.run()
                
                # Store results
                self.results['zero_line'][rho] = zero_results
                self.results['infinite_line'][rho] = infinite_results
                
            self.logger.log_event('analysis_complete', {'rho': rho})
            
    def generate_visualizations(self):
        """Generate paper figures and interactive visualizations."""
        with self.logger.time_operation('visualization'):
            # Extract results for plotting
            rho_values = sorted(self.results['zero_line'].keys())
            
            # Figure 4: Average travel distance
            self.plotter.plot_workload_distribution(
                workloads=np.array([
                    [res.workloads for res in self.results['zero_line'].values()],
                    [res.workloads for res in self.results['infinite_line'].values()]
                ]),
                title="Average Command-wide Travel Distance"
            )
            
            # Figure 5: Interdistrict dispatches
            self.plotter.plot_interdistrict_fractions(
                fractions=np.array([
                    [res.interdistrict_fractions for res in self.results['zero_line'].values()],
                    [res.interdistrict_fractions for res in self.results['infinite_line'].values()]
                ]),
                rho_values=rho_values
            )
            
            # Figure 6: Workload distributions
            self.plotter.plot_performance_comparison(
                zero_line_results=self.results['zero_line'],
                infinite_line_results=self.results['infinite_line'],
                rho_values=rho_values
            )
            
            # Save figures
            self.plotter.save_figures(
                self.output_dir / 'figures' / 'paper_replication',
                formats=['pdf', 'png']
            )
            
            # Create interactive dashboard
            dashboard = self.interactive_viz.create_performance_dashboard({
                'zero_line': self.results['zero_line'],
                'infinite_line': self.results['infinite_line'],
                'rho_values': rho_values
            })
            
            # Save interactive visualization
            self.interactive_viz.save_figure(
                dashboard,
                self.output_dir / 'figures' / 'interactive_dashboard'
            )
            
    def generate_report(self):
        """Generate analysis report."""
        with self.logger.time_operation('report_generation'):
            report_path = self.output_dir / 'report.md'
            
            with open(report_path, 'w') as f:
                f.write("# Linear Command Analysis Report\n\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                
                # System configuration
                f.write("## System Configuration\n")
                f.write("- Nine districts (N=9)\n")
                f.write("- Eighteen atoms (J=18)\n")
                f.write("- Linear command structure\n\n")
                
                # Performance summary
                f.write("## Performance Summary\n")
                for rho in sorted(self.results['zero_line'].keys()):
                    f.write(f"\n### System Utilization ρ = {rho:.1f}\n")
                    
                    # Zero-line results
                    zero_res = self.results['zero_line'][rho]
                    f.write("\nZero-line Capacity Model:\n")
                    f.write(f"- Mean workload: {np.mean(zero_res.workloads):.3f}\n")
                    f.write(f"- Workload imbalance: {np.max(zero_res.workloads) - np.min(zero_res.workloads):.3f}\n")
                    
                    # Infinite-line results
                    inf_res = self.results['infinite_line'][rho]
                    f.write("\nInfinite-line Capacity Model:\n")
                    f.write(f"- Mean workload: {np.mean(inf_res.workloads):.3f}\n")
                    f.write(f"- Workload imbalance: {np.max(inf_res.workloads) - np.min(inf_res.workloads):.3f}\n")
                    if inf_res.queue_metrics:
                        f.write(f"- Mean queue length: {inf_res.queue_metrics['mean_queue_length']:.3f}\n")
                        
        self.logger.log_event('report_generated', {'path': str(report_path)})
        
def main():
    """Run the linear command example."""
    example = LinearCommandExample()
    
    try:
        # Run analysis
        example.run_analysis()
        
        # Generate visualizations
        example.generate_visualizations()
        
        # Generate report
        example.generate_report()
        
        print("Analysis completed successfully. Check the 'results' directory for outputs.")
        
    except Exception as e:
        example.logger.log_error_with_context(e, {
            'phase': 'main_execution',
            'timestamp': datetime.now().isoformat()
        })
        raise

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    print(sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
    main()