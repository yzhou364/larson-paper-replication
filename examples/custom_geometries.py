"""
Example demonstrating custom geometry configurations for the hypercube model.
Includes grid layouts, irregular districts, and custom travel times.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from src.models.zero_line_model import ZeroLineModel
from src.utils.config import ConfigManager, ModelConfig, SystemConfig, GeometryConfig
from src.utils.logging_utils import ModelLogger
from src.visualization.figures import HypercubePlotter
from src.geometry.atoms import AtomManager
from src.geometry.districts import DistrictManager
from src.geometry.travel_time import TravelTimeCalculator

class CustomGeometryExample:
    """Demonstrates different geometric configurations of the hypercube model."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize example.
        
        Args:
            output_dir (Optional[str]): Output directory
        """
        self.output_dir = Path(output_dir or 'results/custom_geometries')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = ModelLogger(name="CustomGeometries")
        self.logger.setup_file_logging(self.output_dir / 'logs')
        self.plotter = HypercubePlotter(style='paper')
        
    def create_grid_configuration(self, rows: int, cols: int) -> Tuple[AtomManager, DistrictManager]:
        """Create grid-based geometry.
        
        Args:
            rows (int): Number of rows
            cols (int): Number of columns
            
        Returns:
            Tuple[AtomManager, DistrictManager]: Configured managers
        """
        N = rows * cols  # Number of districts
        J = N * 4       # 4 atoms per district
        
        # Create atoms in grid layout
        atom_manager = AtomManager(J)
        atoms = atom_manager.create_grid_atoms(
            width=cols * 1.0,
            height=rows * 1.0,
            rows=rows * 2,  # 2x2 atoms per district
            cols=cols * 2
        )
        
        # Create districts
        district_manager = DistrictManager(N, atom_manager)
        districts = district_manager.create_grid_districts(rows, cols)
        
        return atom_manager, district_manager
        
    def create_irregular_configuration(self, N: int) -> Tuple[AtomManager, DistrictManager]:
        """Create irregular district configuration.
        
        Args:
            N (int): Number of districts
            
        Returns:
            Tuple[AtomManager, DistrictManager]: Configured managers
        """
        J = N * 3  # 3 atoms per district on average
        
        # Create atoms with varying sizes
        atom_manager = AtomManager(J)
        areas = np.random.uniform(0.5, 1.5, J)
        atoms = atom_manager.create_custom_atoms(areas)
        
        # Create irregular districts
        district_manager = DistrictManager(N, atom_manager)
        districts = district_manager.create_optimized_districts(
            objective='workload_balance',
            max_iterations=100
        )
        
        return atom_manager, district_manager
        
    def create_custom_travel_times(self, atom_manager: AtomManager,
                                 traffic_zones: List[Dict]) -> np.ndarray:
        """Create custom travel time matrix with traffic zones.
        
        Args:
            atom_manager (AtomManager): Atom management system
            traffic_zones (List[Dict]): Traffic zone definitions
            
        Returns:
            numpy.ndarray: Custom travel time matrix
        """
        calculator = TravelTimeCalculator(atom_manager)
        
        # Create base travel times
        travel_times = calculator.get_atom_distances()
        
        # Apply traffic zone modifiers
        for zone in traffic_zones:
            atoms = zone['atoms']
            factor = zone['congestion_factor']
            for i in atoms:
                for j in atoms:
                    if i != j:
                        travel_times[i,j] *= factor
                        
        return travel_times
        
    def run_grid_analysis(self):
        """Run analysis with grid configuration."""
        with self.logger.time_operation('grid_analysis'):
            # Create 3x3 grid configuration
            atom_manager, district_manager = self.create_grid_configuration(3, 3)
            
            # Setup model configuration
            config = ModelConfig(
                system=SystemConfig(
                    N=9,    # 3x3 districts
                    J=36,   # 4 atoms per district
                    lambda_rate=4.5,  # ρ = 0.5
                    mu_rate=1.0
                ),
                geometry=GeometryConfig(
                    is_grid=True,
                    rows=3,
                    cols=3
                ),
                computation=None,  # Use defaults
                output=None       # Use defaults
            )
            
            # Run model
            model = ZeroLineModel(config)
            model.set_geometry(atom_manager, district_manager)
            results = model.run()
            
            # Visualize results
            self._plot_grid_results(results, atom_manager, district_manager)
            
    def run_irregular_analysis(self):
        """Run analysis with irregular configuration."""
        with self.logger.time_operation('irregular_analysis'):
            # Create irregular configuration
            atom_manager, district_manager = self.create_irregular_configuration(8)
            
            # Define traffic zones
            traffic_zones = [
                {
                    'atoms': [0, 1, 2, 3],
                    'congestion_factor': 1.5  # High traffic area
                },
                {
                    'atoms': [10, 11, 12],
                    'congestion_factor': 1.3  # Medium traffic area
                }
            ]
            
            # Create custom travel times
            travel_times = self.create_custom_travel_times(
                atom_manager, traffic_zones
            )
            
            # Setup model configuration
            config = ModelConfig(
                system=SystemConfig(
                    N=8,
                    J=24,
                    lambda_rate=4.0,
                    mu_rate=1.0
                ),
                geometry=GeometryConfig(
                    is_grid=False
                ),
                computation=None,
                output=None
            )
            
            # Run model
            model = ZeroLineModel(config)
            model.set_geometry(atom_manager, district_manager)
            model.set_travel_times(travel_times)
            results = model.run()
            
            # Visualize results
            self._plot_irregular_results(results, atom_manager, district_manager)
            
    def _plot_grid_results(self, results: Dict, 
                          atom_manager: AtomManager,
                          district_manager: DistrictManager):
        """Plot results for grid configuration."""
        fig = plt.figure(figsize=(15, 5))
        
        # District configuration
        ax1 = fig.add_subplot(131)
        self.plotter.plot_district_map(district_manager, atom_manager, ax=ax1)
        ax1.set_title('District Configuration')
        
        # Workload distribution
        ax2 = fig.add_subplot(132)
        self.plotter.plot_workload_distribution(results.workloads, ax=ax2)
        ax2.set_title('Unit Workloads')
        
        # Travel time heatmap
        ax3 = fig.add_subplot(133)
        self.plotter.plot_travel_times(results.travel_times, ax=ax3)
        ax3.set_title('Travel Times')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'grid_analysis.pdf')
        
    def _plot_irregular_results(self, results: Dict,
                              atom_manager: AtomManager,
                              district_manager: DistrictManager):
        """Plot results for irregular configuration."""
        fig = plt.figure(figsize=(15, 10))
        
        # District configuration
        ax1 = fig.add_subplot(221)
        self.plotter.plot_district_map(district_manager, atom_manager, ax=ax1)
        ax1.set_title('Irregular Districts')
        
        # Workload distribution
        ax2 = fig.add_subplot(222)
        self.plotter.plot_workload_distribution(results.workloads, ax=ax2)
        ax2.set_title('Unit Workloads')
        
        # Travel time heatmap
        ax3 = fig.add_subplot(223)
        self.plotter.plot_travel_times(results.travel_times, ax=ax3)
        ax3.set_title('Travel Times with Traffic Zones')
        
        # Coverage analysis
        ax4 = fig.add_subplot(224)
        coverage = district_manager.compute_coverage_metrics()
        self.plotter.plot_coverage_analysis(coverage, ax=ax4)
        ax4.set_title('District Coverage')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'irregular_analysis.pdf')
        
def main():
    """Run custom geometry examples."""
    example = CustomGeometryExample()
    
    try:
        # Run grid analysis
        example.run_grid_analysis()
        
        # Run irregular analysis
        example.run_irregular_analysis()
        
        print("Analysis completed. Results saved in 'results/custom_geometries'")
        
    except Exception as e:
        example.logger.log_error_with_context(e, {
            'phase': 'custom_geometry_analysis'
        })
        raise

if __name__ == "__main__":
    main()