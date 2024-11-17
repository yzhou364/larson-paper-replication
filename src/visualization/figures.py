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
        self.current_figure = None
        
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
    
    def plot_workload_distribution(self, workloads: np.ndarray,
                                title: str = "Unit Workload Distribution") -> Tuple[plt.Figure, plt.Axes]:
        """Plot unit workload distribution.
        
        Args:
            workloads (numpy.ndarray): Unit workloads
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        workloads = np.asarray(workloads).flatten()  # Ensure 1D numpy array
        
        fig, ax = plt.subplots()
        self.current_figure = fig
        
        # Bar plot
        x = np.arange(1, len(workloads) + 1)
        bars = ax.bar(x, workloads, color='skyblue', alpha=0.7)
        
        # Color bars based on workload
        min_val, max_val = workloads.min(), workloads.max()
        norm = plt.Normalize(min_val, max_val)
        for bar, workload in zip(bars, workloads):
            bar.set_color(plt.cm.YlOrRd(norm(workload)))
            
        # Add mean line
        mean_workload = np.mean(workloads)
        ax.axhline(mean_workload, color='red', linestyle='--',
                  label=f'Mean: {mean_workload:.2f}')
        
        # Format plot
        ax.set_xlabel("Unit Number")
        ax.set_ylabel("Workload")
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Set x-axis ticks
        ax.set_xticks(x)
        ax.set_xticklabels([f'Unit {i}' for i in range(1, len(workloads) + 1)], 
                          rotation=45)
        
        # Set y-axis limits
        ax.set_ylim(0, max(workloads) * 1.1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_performance_comparison(self, zero_line_results: Dict,
                                 infinite_line_results: Dict,
                                 rho_values: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot performance comparison between queue types.
        
        Args:
            zero_line_results (Dict): Zero-line capacity results
            infinite_line_results (Dict): Infinite-line capacity results
            rho_values (numpy.ndarray): System utilization values
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.current_figure = fig
        
        # Prepare data
        zero_workloads = [np.mean(zero_line_results[rho]['workloads']) for rho in rho_values]
        inf_workloads = [np.mean(infinite_line_results[rho]['workloads']) for rho in rho_values]
        
        zero_times = [zero_line_results[rho]['travel_times']['average'] for rho in rho_values]
        inf_times = [infinite_line_results[rho]['travel_times']['average'] for rho in rho_values]
        
        zero_inter = [np.mean(zero_line_results[rho]['interdistrict_fraction']) for rho in rho_values]
        inf_inter = [np.mean(infinite_line_results[rho]['interdistrict_fraction']) for rho in rho_values]
        
        # Plot workload comparison
        ax1.plot(rho_values, zero_workloads, 'b-o', label='Zero-line')
        ax1.plot(rho_values, inf_workloads, 'r-o', label='Infinite-line')
        ax1.set_xlabel('System Utilization (ρ)')
        ax1.set_ylabel('Average Workload')
        ax1.set_title('Workload Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot travel time comparison
        ax2.plot(rho_values, zero_times, 'b-o', label='Zero-line')
        ax2.plot(rho_values, inf_times, 'r-o', label='Infinite-line')
        ax2.set_xlabel('System Utilization (ρ)')
        ax2.set_ylabel('Average Travel Time')
        ax2.set_title('Travel Time Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot interdistrict response comparison
        ax3.plot(rho_values, zero_inter, 'b-o', label='Zero-line')
        ax3.plot(rho_values, inf_inter, 'r-o', label='Infinite-line')
        ax3.set_xlabel('System Utilization (ρ)')
        ax3.set_ylabel('Interdistrict Response Fraction')
        ax3.set_title('Interdistrict Response Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot queue metrics (infinite-line only)
        queue_lengths = []
        queue_times = []
        for rho in rho_values:
            if 'queue_metrics' in infinite_line_results[rho]:
                metrics = infinite_line_results[rho]['queue_metrics']
                queue_lengths.append(metrics['expected_queue_length'])
                queue_times.append(metrics['expected_wait_time'])
        
        if queue_lengths and queue_times:
            ax4.plot(rho_values[:len(queue_lengths)], queue_lengths, 'g-o',
                    label='Queue Length')
            ax4.plot(rho_values[:len(queue_times)], queue_times, 'm-o',
                    label='Wait Time')
            ax4.set_xlabel('System Utilization (ρ)')
            ax4.set_ylabel('Queue Metrics')
            ax4.set_title('Queue Performance (Infinite-line)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        return fig, ((ax1, ax2), (ax3, ax4))
    
    def save_figures(self, filepath: str, formats: List[str] = ['pdf', 'png']):
        """Save current figure in specified formats.
        
        Args:
            filepath (str): Base filepath for saving
            formats (List[str]): Output formats
        """
        if self.current_figure is None:
            return
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            save_path = filepath.with_suffix(f'.{fmt}')
            self.current_figure.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                format=fmt
            )
    
    def _set_style(self):
        """Set matplotlib style based on output type."""
        base_settings = {
            'axes.grid': True,
            'grid.linestyle': ':',
            'grid.alpha': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
        
        if self.style == 'paper':
            plt.rcParams.update({
                **base_settings,
                'figure.figsize': (6, 4),
                'font.size': 10,
                'axes.labelsize': 10,
                'axes.titlesize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'lines.linewidth': 1.5,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                **base_settings,
                'figure.figsize': (10, 6),
                'font.size': 14,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'lines.linewidth': 2.0,
            })
        else:  # interactive
            plt.rcParams.update({
                **base_settings,
                'figure.figsize': (8, 5),
                'font.size': 12,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'lines.linewidth': 1.5,
            })