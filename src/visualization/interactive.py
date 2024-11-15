import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

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
        
    def create_workload_dashboard(self, workloads: np.ndarray, 
                                rho_values: np.ndarray,
                                district_info: Optional[Dict] = None) -> go.Figure:
        """Create interactive workload analysis dashboard.
        
        Args:
            workloads (numpy.ndarray): Unit workloads over time
            rho_values (numpy.ndarray): System utilization values
            district_info (Optional[Dict]): District information
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Unit Workload Distribution',
                'Workload Over Time',
                'Workload Imbalance',
                'District Analysis'
            )
        )
        
        # Unit workload distribution
        fig.add_trace(
            go.Bar(
                x=[f'Unit {i+1}' for i in range(len(workloads))],
                y=workloads[-1],  # Latest workloads
                name='Current Workload'
            ),
            row=1, col=1
        )
        
        # Workload over time
        for i in range(len(workloads[0])):
            fig.add_trace(
                go.Scatter(
                    x=rho_values,
                    y=workloads[:, i],
                    name=f'Unit {i+1}',
                    mode='lines+markers'
                ),
                row=1, col=2
            )
            
        # Workload imbalance
        imbalance = np.max(workloads, axis=1) - np.min(workloads, axis=1)
        fig.add_trace(
            go.Scatter(
                x=rho_values,
                y=imbalance,
                name='Workload Imbalance',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # District analysis if available
        if district_info is not None:
            fig.add_trace(
                go.Box(
                    y=list(district_info.values()),
                    name='District Workloads'
                ),
                row=2, col=2
            )
            
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            showlegend=True,
            title_text="Workload Analysis Dashboard",
            title_font_size=self.config.title_font_size
        )
        
        return fig
    
    def create_travel_time_heatmap(self, travel_times: np.ndarray,
                                 district_boundaries: Optional[List] = None) -> go.Figure:
        """Create interactive travel time heatmap.
        
        Args:
            travel_times (numpy.ndarray): Travel time matrix
            district_boundaries (Optional[List]): District boundary indices
            
        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=travel_times,
            colorscale=self.config.colorscale,
            text=np.round(travel_times, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        
        # Add district boundaries if provided
        if district_boundaries:
            for boundary in district_boundaries:
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=boundary - 0.5,
                    x1=len(travel_times) - 0.5,
                    y1=boundary - 0.5,
                    line=dict(color="black", width=2)
                )
                
        fig.update_layout(
            title="Travel Time Matrix",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            xaxis_title="To Atom",
            yaxis_title="From Atom"
        )
        
        return fig
    
    def create_system_animation(self, states: List[np.ndarray],
                              times: List[float]) -> go.Figure:
        """Create animated visualization of system states.
        
        Args:
            states (List[numpy.ndarray]): System states over time
            times (List[float]): Time points
            
        Returns:
            plotly.graph_objects.Figure: Animated visualization
        """
        frames = []
        for state, time in zip(states, times):
            frame = go.Frame(
                data=[go.Heatmap(
                    z=state,
                    colorscale=self.config.colorscale,
                    showscale=False
                )],
                name=f't={time:.2f}'
            )
            frames.append(frame)
            
        fig = go.Figure(
            data=[go.Heatmap(z=states[0], colorscale=self.config.colorscale)],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play',
                     'method': 'animate',
                     'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                   'fromcurrent': True}]},
                    {'label': 'Pause',
                     'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                     'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Time: '},
                'steps': [{'args': [[f't={t:.2f}'], {'frame': {'duration': 0, 'redraw': True},
                                                   'mode': 'immediate'}],
                          'label': f'{t:.2f}',
                          'method': 'animate'} for t in times]
            }]
        )
        
        return fig
    
    def create_performance_dashboard(self, results: Dict) -> go.Figure:
        """Create interactive performance analysis dashboard.
        
        Args:
            results (Dict): Model results
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Response Time Distribution',
                'Workload Balance',
                'Coverage Analysis',
                'Queue Statistics'
            )
        )
        
        # Response time distribution
        if 'response_times' in results:
            fig.add_trace(
                go.Histogram(
                    x=results['response_times'],
                    name='Response Times',
                    nbinsx=30
                ),
                row=1, col=1
            )
            
        # Workload balance
        if 'workloads' in results:
            fig.add_trace(
                go.Box(
                    y=results['workloads'],
                    name='Unit Workloads'
                ),
                row=1, col=2
            )
            
        # Coverage analysis
        if 'coverage' in results:
            fig.add_trace(
                go.Bar(
                    x=list(results['coverage'].keys()),
                    y=list(results['coverage'].values()),
                    name='Coverage'
                ),
                row=2, col=1
            )
            
        # Queue statistics
        if 'queue_stats' in results:
            fig.add_trace(
                go.Scatter(
                    x=results['queue_stats']['times'],
                    y=results['queue_stats']['lengths'],
                    mode='lines',
                    name='Queue Length'
                ),
                row=2, col=2
            )
            
        fig.update_layout(
            height=800,
            title_text="System Performance Dashboard",
            showlegend=True
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str,
                   format: str = 'html'):
        """Save interactive figure.
        
        Args:
            fig (plotly.graph_objects.Figure): Figure to save
            filename (str): Output filename
            format (str): Output format ('html' or 'json')
        """
        if format == 'html':
            fig.write_html(f"{filename}.html")
        elif format == 'json':
            fig.write_json(f"{filename}.json")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def update_figure_style(self, fig: go.Figure):
        """Update figure style settings.
        
        Args:
            fig (plotly.graph_objects.Figure): Figure to update
        """
        fig.update_layout(
            template=self.config.template,
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.axis_font_size),
            title_font_size=self.config.title_font_size,
            legend_font_size=self.config.legend_font_size,
            showgrid=self.config.show_grid
        )