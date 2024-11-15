import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import logging
from enum import Enum

class ModelType(Enum):
    """Types of hypercube models."""
    ZERO_LINE = "zero_line"
    INFINITE_LINE = "infinite_line"

class DispatchType(Enum):
    """Types of dispatch policies."""
    MCM = "mcm"
    DISTRICT = "district"
    WORKLOAD = "workload"

@dataclass
class SystemConfig:
    """System configuration parameters."""
    N: int  # Number of units
    J: int  # Number of atoms
    lambda_rate: float  # Arrival rate
    mu_rate: float = 1.0  # Service rate
    model_type: ModelType = ModelType.ZERO_LINE
    dispatch_type: DispatchType = DispatchType.MCM

@dataclass
class GeometryConfig:
    """Geometry configuration parameters."""
    district_length: float = 1.0
    is_grid: bool = False
    rows: Optional[int] = None
    cols: Optional[int] = None
    use_manhattan_distance: bool = False
    atom_areas: Optional[Dict[int, float]] = None

@dataclass
class ComputationConfig:
    """Computation configuration parameters."""
    max_iterations: int = 1000
    tolerance: float = 1e-10
    use_cache: bool = True
    parallel_compute: bool = False
    num_threads: int = 1

@dataclass
class OutputConfig:
    """Output configuration parameters."""
    save_path: str = "results"
    save_format: str = "json"
    plot_results: bool = True
    generate_report: bool = True
    verbose: bool = True

@dataclass
class ModelConfig:
    """Complete model configuration."""
    system: SystemConfig
    geometry: GeometryConfig
    computation: ComputationConfig
    output: OutputConfig
    custom_params: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """Manages model configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path (Optional[Union[str, Path]]): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = None
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file.
        
        Args:
            config_path (Union[str, Path]): Path to configuration file
        """
        config_path = Path(config_path)
        
        try:
            if config_path.suffix == '.yaml':
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path) as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
                
            self.config = self._create_config_from_dict(config_dict)
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
            
    def save_config(self, save_path: Union[str, Path]):
        """Save current configuration to file.
        
        Args:
            save_path (Union[str, Path]): Path to save configuration
        """
        if self.config is None:
            raise ValueError("No configuration to save")
            
        save_path = Path(save_path)
        config_dict = self._config_to_dict()
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.yaml':
                with open(save_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif save_path.suffix == '.json':
                with open(save_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
            else:
                raise ValueError(f"Unsupported config format: {save_path.suffix}")
                
            self.logger.info(f"Saved configuration to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise
            
    def _create_config_from_dict(self, config_dict: Dict) -> ModelConfig:
        """Create configuration objects from dictionary.
        
        Args:
            config_dict (Dict): Configuration dictionary
            
        Returns:
            ModelConfig: Created configuration
        """
        # Create system config
        system_config = SystemConfig(
            N=config_dict['system']['N'],
            J=config_dict['system']['J'],
            lambda_rate=config_dict['system']['lambda_rate'],
            mu_rate=config_dict['system'].get('mu_rate', 1.0),
            model_type=ModelType(config_dict['system'].get('model_type', 'zero_line')),
            dispatch_type=DispatchType(config_dict['system'].get('dispatch_type', 'mcm'))
        )
        
        # Create geometry config
        geometry_config = GeometryConfig(
            district_length=config_dict['geometry'].get('district_length', 1.0),
            is_grid=config_dict['geometry'].get('is_grid', False),
            rows=config_dict['geometry'].get('rows'),
            cols=config_dict['geometry'].get('cols'),
            use_manhattan_distance=config_dict['geometry'].get('use_manhattan_distance', False),
            atom_areas=config_dict['geometry'].get('atom_areas')
        )
        
        # Create computation config
        computation_config = ComputationConfig(
            max_iterations=config_dict['computation'].get('max_iterations', 1000),
            tolerance=config_dict['computation'].get('tolerance', 1e-10),
            use_cache=config_dict['computation'].get('use_cache', True),
            parallel_compute=config_dict['computation'].get('parallel_compute', False),
            num_threads=config_dict['computation'].get('num_threads', 1)
        )
        
        # Create output config
        output_config = OutputConfig(
            save_path=config_dict['output'].get('save_path', 'results'),
            save_format=config_dict['output'].get('save_format', 'json'),
            plot_results=config_dict['output'].get('plot_results', True),
            generate_report=config_dict['output'].get('generate_report', True),
            verbose=config_dict['output'].get('verbose', True)
        )
        
        # Create complete model config
        return ModelConfig(
            system=system_config,
            geometry=geometry_config,
            computation=computation_config,
            output=output_config,
            custom_params=config_dict.get('custom_params', {})
        )
        
    def _config_to_dict(self) -> Dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dict: Configuration dictionary
        """
        if self.config is None:
            return {}
            
        return {
            'system': asdict(self.config.system),
            'geometry': asdict(self.config.geometry),
            'computation': asdict(self.config.computation),
            'output': asdict(self.config.output),
            'custom_params': self.config.custom_params
        }
        
    def validate_config(self) -> bool:
        """Validate current configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        if self.config is None:
            return False
            
        try:
            # Validate system config
            if self.config.system.N <= 0 or self.config.system.J <= 0:
                raise ValueError("N and J must be positive")
                
            if self.config.system.lambda_rate <= 0 or self.config.system.mu_rate <= 0:
                raise ValueError("Rates must be positive")
                
            # Validate geometry config
            if self.config.geometry.is_grid:
                if not (self.config.geometry.rows and self.config.geometry.cols):
                    raise ValueError("Grid configuration requires rows and cols")
                    
            # Validate computation config
            if self.config.computation.max_iterations <= 0:
                raise ValueError("max_iterations must be positive")
                
            if self.config.computation.tolerance <= 0:
                raise ValueError("tolerance must be positive")
                
            # Validate output config
            save_path = Path(self.config.output.save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
        
    def update_config(self, updates: Dict):
        """Update configuration parameters.
        
        Args:
            updates (Dict): Configuration updates
        """
        if self.config is None:
            raise ValueError("No configuration to update")
            
        config_dict = self._config_to_dict()
        
        # Update configuration recursively
        def update_dict(d: Dict, u: Dict):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        updated_dict = update_dict(config_dict, updates)
        self.config = self._create_config_from_dict(updated_dict)
        self.logger.info("Configuration updated")
        
    def get_config(self) -> ModelConfig:
        """Get current configuration.
        
        Returns:
            ModelConfig: Current configuration
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        return self.config
    

    def setup_default_config(self) -> ModelConfig:
        """Create and return default configuration.
        
        Returns:
            ModelConfig: Default configuration
        """
        # Create default system configuration
        system_config = SystemConfig(
            N=9,  # 9 districts as in paper
            J=18,  # 18 atoms
            lambda_rate=4.5,  # Default arrival rate
            mu_rate=1.0,  # Default service rate
            model_type=ModelType.ZERO_LINE,
            dispatch_type=DispatchType.MCM
        )
        
        # Create default geometry configuration
        geometry_config = GeometryConfig(
            district_length=1.0,
            is_grid=False,  # Linear configuration as in paper
            use_manhattan_distance=False
        )
        
        # Create default computation configuration
        computation_config = ComputationConfig(
            max_iterations=1000,
            tolerance=1e-10,
            use_cache=True,
            parallel_compute=False,
            num_threads=1
        )
        
        # Create default output configuration
        output_config = OutputConfig(
            save_path="results",
            save_format="json",
            plot_results=True,
            generate_report=True,
            verbose=True
        )
        
        # Create complete model configuration
        self.config = ModelConfig(
            system=system_config,
            geometry=geometry_config,
            computation=computation_config,
            output=output_config
        )
        
        return self.config