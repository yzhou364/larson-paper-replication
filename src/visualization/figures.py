"""
Visualization tools for hypercube model results using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from pathlib import Path

class HypercubePlotter:
    """Creates publication-quality plots for hypercube model results."""
    
    def __init__(self, style: str = 'paper'):
        """Initialize plotter with style settings.
        
        Args:
            style (str): Plot style ('paper', 'presentation', or 'interactive')
        """
        self.style = style
        
        # Set default style
        plt.style.use('default')
        sns.set_theme()  # Apply seaborn defaults
        
        self._set_style()
        
        # Color schemes
        self.colors = {
            'districts': plt.cm.Set3(np.linspace(0, 1, 12)),
            'workload': sns.color_palette("YlOrRd", 8),
            'travel_time': sns.color_palette("YlGnBu", 8),
            'interdistrict': sns.color_palette("RdPu", 8)
        }
        
    def _set_style(self):
        """Set matplotlib style based on output type."""
        if self.style == 'paper':
            plt.rcParams.update({
                'figure.figsize': (6, 4),
                'font.size': 10,
                'axes.labelsize': 10,
                'axes.titlesize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'lines.linewidth': 1.5,
                'axes.grid': True,
                'grid.linestyle': '--',
                'grid.alpha': 0.7,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'figure.figsize': (10, 6),
                'font.size': 14,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'lines.linewidth': 2.0,
                'axes.grid': True,
                'grid.linestyle': '--',
                'grid.alpha': 0.7,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'figure.dpi': 150,
                'savefig.dpi': 150
            })
        else:  # interactive
            plt.rcParams.update({
                'figure.figsize': (8, 5),
                'font.size': 12,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'lines.linewidth': 1.5,
                'axes.grid': True,
                'grid.linestyle': ':',
                'grid.alpha': 0.5
            })

            
    def plot_workload_distribution(self, workloads: np.ndarray,
                                title: str = "Unit Workload Distribution") -> Tuple:
        """Plot unit workload distribution.
        
        Args:
            workloads (numpy.ndarray): Unit workloads
            title (str): Plot title
            
        Returns:
            Tuple: Figure and axes objects
        """
        fig, ax = plt.subplots()
        
        # Bar plot
        bars = ax.bar(range(1, len(workloads) + 1), workloads)
        
        # Color bars based on workload
        norm = plt.Normalize(workloads.min(), workloads.max())
        for bar, workload in zip(bars, workloads):
            bar.set_color(self.colors['workload'][int(norm(workload) * 7)])
            
        # Add mean line
        mean_workload = np.mean(workloads)
        ax.axhline(mean_workload, color='red', linestyle='--',
                  label=f'Mean: {mean_workload:.2f}')
        
        ax.set_xlabel("Unit Number")
        ax.set_ylabel("Workload")
        ax.set_title(title)
        ax.legend()
        
        return fig, ax
    
    def plot_travel_times(self, travel_times: np.ndarray,
                        title: str = "Travel Time Distribution") -> Tuple:
        """Plot travel time heatmap.
        
        Args:
            travel_times (numpy.ndarray): Travel time matrix
            title (str): Plot title
            
        Returns:
            Tuple: Figure and axes objects
        """
        fig, ax = plt.subplots()
        
        # Create heatmap
        im = ax.imshow(travel_times, cmap='YlGnBu')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Travel Time')
        
        # Add text annotations
        for i in range(travel_times.shape[0]):
            for j in range(travel_times.shape[1]):
                ax.text(j, i, f'{travel_times[i,j]:.2f}',
                       ha='center', va='center')
                
        ax.set_xlabel("To Atom")
        ax.set_ylabel("From Atom")
        ax.set_title(title)
        
        return fig, ax
    
    def plot_interdistrict_fractions(self, fractions: np.ndarray,
                                   rho_values: np.ndarray) -> Tuple:
        """Plot interdistrict response fractions.
        
        Args:
            fractions (numpy.ndarray): Interdistrict fractions by unit
            rho_values (numpy.ndarray): System utilization values
            
        Returns:
            Tuple: Figure and axes objects
        """
        fig, ax = plt.subplots()
        
        for unit in range(len(fractions)):
            ax.plot(rho_values, fractions[unit], 'o-',
                   label=f'Unit {unit+1}')
            
        ax.set_xlabel("System Utilization (ρ)")
        ax.set_ylabel("Fraction of Interdistrict Responses")
        ax.set_title("Interdistrict Response Patterns")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig, ax
    
    def plot_performance_comparison(self, zero_line_results: Dict,
                                 infinite_line_results: Dict,
                                 rho_values: np.ndarray) -> Tuple:
        """Plot performance comparison between queue types.
        
        Args:
            zero_line_results (Dict): Zero-line capacity results
            infinite_line_results (Dict): Infinite-line capacity results
            rho_values (numpy.ndarray): System utilization values
            
        Returns:
            Tuple: Figure and axes objects
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Travel time comparison
        ax1.plot(rho_values, zero_line_results['travel_times'],
                'b-o', label='Zero-line')
        ax1.plot(rho_values, infinite_line_results['travel_times'],
                'r-o', label='Infinite-line')
        ax1.set_xlabel("System Utilization (ρ)")
        ax1.set_ylabel("Average Travel Time")
        ax1.set_title("Travel Time Comparison")
        ax1.legend()
        
        # Workload imbalance
        ax2.plot(rho_values, zero_line_results['workload_imbalance'],
                'b-o', label='Zero-line')
        ax2.plot(rho_values, infinite_line_results['workload_imbalance'],
                'r-o', label='Infinite-line')
        ax2.set_xlabel("System Utilization (ρ)")
        ax2.set_ylabel("Workload Imbalance")
        ax2.set_title("Workload Imbalance Comparison")
        ax2.legend()
        
        # Interdistrict response fraction
        ax3.plot(rho_values, zero_line_results['interdistrict_fraction'],
                'b-o', label='Zero-line')
        ax3.plot(rho_values, infinite_line_results['interdistrict_fraction'],
                'r-o', label='Infinite-line')
        ax3.set_xlabel("System Utilization (ρ)")
        ax3.set_ylabel("Interdistrict Response Fraction")
        ax3.set_title("Interdistrict Response Comparison")
        ax3.legend()
        
        # Queue metrics (infinite-line only)
        if 'queue_metrics' in infinite_line_results:
            metrics = infinite_line_results['queue_metrics']
            ax4.plot(rho_values, metrics['mean_queue_length'],
                    'g-o', label='Mean Queue Length')
            ax4.plot(rho_values, metrics['mean_wait_time'],
                    'm-o', label='Mean Wait Time')
            ax4.set_xlabel("System Utilization (ρ)")
            ax4.set_ylabel("Queue Metrics")
            ax4.set_title("Queue Performance (Infinite-line)")
            ax4.legend()
            
        plt.tight_layout()
        return fig, ((ax1, ax2), (ax3, ax4))
    
    def plot_district_map(self, district_manager: 'DistrictManager',
                        atom_manager: 'AtomManager',
                        title: str = "District Configuration") -> Tuple:
        """Plot district configuration.
        
        Args:
            district_manager: District management system
            atom_manager: Atom management system
            title (str): Plot title
            
        Returns:
            Tuple: Figure and axes objects
        """
        fig, ax = plt.subplots()
        
        # Plot atoms colored by district
        for atom_id, atom in atom_manager.atoms.items():
            district = district_manager.assignments[atom_id]
            color = self.colors['districts'][district % 12]
            
            # Create circle for atom
            radius = np.sqrt(atom.area / np.pi)
            circle = plt.Circle((atom.center.x, atom.center.y),
                              radius, color=color, alpha=0.6)
            ax.add_patch(circle)
            
        # Plot district centers
        for district in district_manager.districts.values():
            ax.plot(district.center.x, district.center.y, 'k*',
                   markersize=10, label=f'District {district.id + 1}')
            
        ax.set_aspect('equal')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        ax.legend()
        
        return fig, ax
    
    def save_figures(self, output_dir: str, formats: List[str] = ['pdf', 'png']):
        """Save all open figures.
        
        Args:
            output_dir (str): Output directory
            formats (List[str]): Output formats
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(plt.get_fignums()):
            for fmt in formats:
                plt.figure(fig)
                plt.savefig(output_dir / f'figure_{i+1}.{fmt}',
                          bbox_inches='tight', dpi=300)
                
    def close_all(self):
        """Close all open figures."""
        plt.close('all')