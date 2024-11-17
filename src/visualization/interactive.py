"""
Interactive visualization tools for hypercube model results using plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class InteractiveConfig:
    """Configuration for interactive visualizations."""
    width: int = 800
    height: int = 600
    template: str = 'plotly_white'
    colorscale: str = 'Viridis'
    show_grid: bool = True
    title_font_size: int = 20
    axis_font_size: int = 14
    legend_font_size: int = 12

class InteractiveVisualizer:
    """Creates interactive visualizations for hypercube model results."""
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        """Initialize interactive visualizer.
        
        Args:
            config (Optional[InteractiveConfig]): Visualization configuration
        """
        self.config = config or InteractiveConfig()
    
    def create_performance_dashboard(self, results: Dict) -> go.Figure:
        """Create interactive performance analysis dashboard.
        
        Args:
            results (Dict): Model results containing zero_line and infinite_line data
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        # Extract rho values
        rho_values = sorted(results['zero_line'].keys())
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Response Time Distribution',
                'Workload Balance',
                'Coverage Analysis',
                'Queue Statistics'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot response time comparison
        zero_times = [results['zero_line'][rho]['travel_times']['average'] for rho in rho_values]
        inf_times = [results['infinite_line'][rho]['travel_times']['average'] for rho in rho_values]
        
        fig.add_trace(
            go.Scatter(x=rho_values, y=zero_times, name='Zero-line Response Time',
                      mode='lines+markers', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rho_values, y=inf_times, name='Infinite-line Response Time',
                      mode='lines+markers', line=dict(color='red')),
            row=1, col=1
        )
        
        # Plot workload balance
        zero_workloads = [np.mean(results['zero_line'][rho]['workloads']) for rho in rho_values]
        inf_workloads = [np.mean(results['infinite_line'][rho]['workloads']) for rho in rho_values]
        
        fig.add_trace(
            go.Scatter(x=rho_values, y=zero_workloads, name='Zero-line Workload',
                      mode='lines+markers', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=rho_values, y=inf_workloads, name='Infinite-line Workload',
                      mode='lines+markers', line=dict(color='red')),
            row=1, col=2
        )
        
        # Plot interdistrict responses
        zero_inter = [np.mean(results['zero_line'][rho]['interdistrict_fraction']) 
                     for rho in rho_values]
        inf_inter = [np.mean(results['infinite_line'][rho]['interdistrict_fraction'])
                    for rho in rho_values]
        
        fig.add_trace(
            go.Scatter(x=rho_values, y=zero_inter, name='Zero-line Interdistrict',
                      mode='lines+markers', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=rho_values, y=inf_inter, name='Infinite-line Interdistrict',
                      mode='lines+markers', line=dict(color='red')),
            row=2, col=1
        )
        
        # Plot queue metrics (infinite-line only)
        queue_lengths = []
        queue_times = []
        for rho in rho_values:
            if 'queue_metrics' in results['infinite_line'][rho]:
                metrics = results['infinite_line'][rho]['queue_metrics']
                queue_lengths.append(metrics.get('expected_queue_length', 0))
                queue_times.append(metrics.get('expected_wait_time', 0))
            else:
                queue_lengths.append(0)
                queue_times.append(0)
        
        fig.add_trace(
            go.Scatter(x=rho_values, y=queue_lengths, name='Queue Length',
                      mode='lines+markers', line=dict(color='green')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=rho_values, y=queue_times, name='Wait Time',
                      mode='lines+markers', line=dict(color='purple')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            template=self.config.template,
            title={
                'text': "System Performance Dashboard",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=self.config.title_font_size)
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=self.config.legend_font_size)
            ),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="System Utilization (ρ)",
            showgrid=self.config.show_grid,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
            tickfont=dict(size=self.config.axis_font_size)
        )
        
        fig.update_yaxes(
            showgrid=self.config.show_grid,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
            tickfont=dict(size=self.config.axis_font_size)
        )
        
        # Update specific y-axis titles
        fig.update_yaxes(title_text="Response Time", row=1, col=1)
        fig.update_yaxes(title_text="Workload", row=1, col=2)
        fig.update_yaxes(title_text="Interdistrict Fraction", row=2, col=1)
        fig.update_yaxes(title_text="Queue Metrics", row=2, col=2)
        
        return fig
    
    def create_workload_heatmap(self, results: Dict) -> go.Figure:
        """Create interactive workload heatmap.
        
        Args:
            results (Dict): Model results
            
        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        # Extract workload data
        rho_values = sorted(results['zero_line'].keys())
        num_units = len(results['zero_line'][rho_values[0]]['workloads'])
        
        # Create workload matrices
        zero_workloads = np.zeros((len(rho_values), num_units))
        inf_workloads = np.zeros((len(rho_values), num_units))
        
        for i, rho in enumerate(rho_values):
            zero_workloads[i] = results['zero_line'][rho]['workloads']
            inf_workloads[i] = results['infinite_line'][rho]['workloads']
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Zero-line Workload', 'Infinite-line Workload')
        )
        
        # Add heatmaps
        fig.add_trace(
            go.Heatmap(z=zero_workloads,
                      x=[f'Unit {i+1}' for i in range(num_units)],
                      y=rho_values,
                      colorscale=self.config.colorscale,
                      name='Zero-line'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(z=inf_workloads,
                      x=[f'Unit {i+1}' for i in range(num_units)],
                      y=rho_values,
                      colorscale=self.config.colorscale,
                      name='Infinite-line'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            title_text="Workload Distribution Heatmap",
            yaxis_title="System Utilization (ρ)",
            yaxis2_title="System Utilization (ρ)"
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: Union[str, Path]):
        """Save interactive figure.
        
        Args:
            fig (plotly.graph_objects.Figure): Figure to save
            filename (Union[str, Path]): Output filename
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as HTML
        fig.write_html(
            f"{filepath}.html",
            include_plotlyjs='cdn',
            full_html=True
        )
        
        # Save as JSON for backup
        fig.write_json(f"{filepath}.json")
    
    def create_queue_analysis_dashboard(self, results: Dict) -> go.Figure:
        """Create interactive queue analysis dashboard.
        
        Args:
            results (Dict): Model results
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Queue Length Distribution',
                'Wait Time Distribution',
                'System Utilization',
                'Performance Metrics'
            )
        )
        
        # Extract data
        rho_values = sorted(results['infinite_line'].keys())
        queue_data = {
            'lengths': [],
            'times': [],
            'utils': []
        }
        
        for rho in rho_values:
            if 'queue_metrics' in results['infinite_line'][rho]:
                metrics = results['infinite_line'][rho]['queue_metrics']
                queue_data['lengths'].append(metrics.get('expected_queue_length', 0))
                queue_data['times'].append(metrics.get('expected_wait_time', 0))
                queue_data['utils'].append(metrics.get('utilization', 0))
        
        # Add traces
        fig.add_trace(
            go.Histogram(x=queue_data['lengths'], name='Queue Length',
                        nbinsx=30),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=queue_data['times'], name='Wait Time',
                        nbinsx=30),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=rho_values, y=queue_data['utils'],
                      name='System Utilization',
                      mode='lines+markers'),
            row=2, col=1
        )
        
        # Add performance box plot
        metrics = ['Queue Length', 'Wait Time', 'Utilization']
        values = [queue_data['lengths'], queue_data['times'], queue_data['utils']]
        
        fig.add_trace(
            go.Box(y=np.array(values).flatten(),
                  x=np.repeat(metrics, [len(v) for v in values]),
                  name='Performance Distribution'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Queue Analysis Dashboard",
            showlegend=True,
            template=self.config.template
        )
        
        return fig