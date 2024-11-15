import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    figure_size: Tuple[int, int] = (10, 6)
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    grid: bool = True
    style: str = 'seaborn'
    color_palette: str = 'deep'
    dpi: int = 300
    save_format: str = 'pdf'

class PlottingUtilities:
    """Utility functions for creating and formatting plots."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize plotting utilities.
        
        Args:
            config (Optional[PlotConfig]): Plot configuration
        """
        self.config = config or PlotConfig()
        self._setup_style()
        
    def _setup_style(self):
        """Apply plot style configuration."""
        plt.style.use(self.config.style)
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'legend.fontsize': self.config.legend_size,
            'figure.dpi': self.config.dpi
        })
        sns.set_palette(self.config.color_palette)
        
    def create_colormap(self, low_color: str, high_color: str, 
                       n_colors: int = 256) -> 'matplotlib.colors.LinearSegmentedColormap':
        """Create custom colormap.
        
        Args:
            low_color (str): Color for minimum values
            high_color (str): Color for maximum values
            n_colors (int): Number of color levels
            
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Created colormap
        """
        return sns.light_palette(high_color, n_colors=n_colors, 
                               as_cmap=True, reverse=False)
        
    def format_axis(self, ax: 'matplotlib.axes.Axes', 
                   title: str = None,
                   xlabel: str = None,
                   ylabel: str = None,
                   xlim: Tuple[float, float] = None,
                   ylim: Tuple[float, float] = None):
        """Format plot axis.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to format
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            xlim (Tuple[float, float]): X-axis limits
            ylim (Tuple[float, float]): Y-axis limits
        """
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if self.config.grid:
            ax.grid(True, linestyle='--', alpha=0.7)
            
    def add_colorbar(self, fig: 'matplotlib.figure.Figure',
                    mappable: 'matplotlib.cm.ScalarMappable',
                    label: str = None):
        """Add colorbar to figure.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to add colorbar to
            mappable (matplotlib.cm.ScalarMappable): Mappable for colorbar
            label (str): Colorbar label
        """
        cbar = fig.colorbar(mappable)
        if label:
            cbar.set_label(label)
            
    def create_subplots(self, rows: int, cols: int, 
                       width_ratios: List[float] = None,
                       height_ratios: List[float] = None) -> Tuple:
        """Create figure with subplots.
        
        Args:
            rows (int): Number of rows
            cols (int): Number of columns
            width_ratios (List[float]): Relative widths of columns
            height_ratios (List[float]): Relative heights of rows
            
        Returns:
            Tuple: Figure and axes array
        """
        fig = plt.figure(figsize=self.config.figure_size)
        gs = fig.add_gridspec(rows, cols, 
                            width_ratios=width_ratios,
                            height_ratios=height_ratios)
        axes = gs.subplots()
        return fig, axes
    
    def add_annotations(self, ax: 'matplotlib.axes.Axes',
                       annotations: List[Dict]):
        """Add text annotations to plot.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to annotate
            annotations (List[Dict]): List of annotation specifications
        """
        for ann in annotations:
            ax.annotate(
                text=ann['text'],
                xy=(ann['x'], ann['y']),
                xytext=ann.get('xytext', None),
                arrowprops=ann.get('arrowprops', None),
                fontsize=ann.get('fontsize', self.config.font_size),
                ha=ann.get('ha', 'center'),
                va=ann.get('va', 'center')
            )
            
    def add_legend(self, ax: 'matplotlib.axes.Axes',
                  loc: str = 'best',
                  title: str = None,
                  ncol: int = 1,
                  bbox_to_anchor: Tuple[float, float] = None):
        """Add and format legend.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to add legend to
            loc (str): Legend location
            title (str): Legend title
            ncol (int): Number of columns
            bbox_to_anchor (Tuple[float, float]): Legend anchor point
        """
        legend = ax.legend(loc=loc, title=title, ncol=ncol, 
                         bbox_to_anchor=bbox_to_anchor)
        if title:
            legend.get_title().set_fontsize(self.config.legend_size)
            
    def save_plot(self, fig: 'matplotlib.figure.Figure',
                 filename: str,
                 formats: List[str] = None):
        """Save plot in specified formats.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to save
            filename (str): Base filename
            formats (List[str]): File formats to save
        """
        formats = formats or [self.config.save_format]
        for fmt in formats:
            fig.savefig(f"{filename}.{fmt}", 
                       bbox_inches='tight',
                       dpi=self.config.dpi)
            
    def create_composite_plot(self, data: List[Dict]) -> Tuple:
        """Create composite plot with multiple elements.
        
        Args:
            data (List[Dict]): Plot specifications
            
        Returns:
            Tuple: Figure and axes
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        for plot_spec in data:
            plot_type = plot_spec.pop('type')
            
            if plot_type == 'line':
                ax.plot(**plot_spec)
            elif plot_type == 'scatter':
                ax.scatter(**plot_spec)
            elif plot_type == 'bar':
                ax.bar(**plot_spec)
            elif plot_type == 'hist':
                ax.hist(**plot_spec)
                
        return fig, ax
    
    @staticmethod
    def add_text_box(ax: 'matplotlib.axes.Axes',
                    text: str,
                    position: Tuple[float, float],
                    boxstyle: str = 'round',
                    facecolor: str = 'white',
                    alpha: float = 0.8):
        """Add text box to plot.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to add text box to
            text (str): Text content
            position (Tuple[float, float]): Box position
            boxstyle (str): Box style
            facecolor (str): Box color
            alpha (float): Box transparency
        """
        props = dict(boxstyle=boxstyle, facecolor=facecolor, alpha=alpha)
        ax.text(position[0], position[1], text, transform=ax.transAxes,
                bbox=props, verticalalignment='top')
                
    @staticmethod
    def format_ticks(ax: 'matplotlib.axes.Axes',
                    xticks: List = None,
                    yticks: List = None,
                    xticklabels: List = None,
                    yticklabels: List = None,
                    rotation: float = 0):
        """Format axis ticks.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to format
            xticks (List): X-axis tick positions
            yticks (List): Y-axis tick positions
            xticklabels (List): X-axis tick labels
            yticklabels (List): Y-axis tick labels
            rotation (float): Label rotation angle
        """
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=rotation)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
            
    @staticmethod
    def set_axis_scale(ax: 'matplotlib.axes.Axes',
                      xscale: str = None,
                      yscale: str = None):
        """Set axis scales.
        
        Args:
            ax (matplotlib.axes.Axes): Axis to modify
            xscale (str): X-axis scale ('linear', 'log', etc.)
            yscale (str): Y-axis scale
        """
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)