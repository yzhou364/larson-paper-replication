from dataclasses import dataclass
from typing import Dict, Optional
from src.utils.config import ModelConfig as GlobalConfig
from src.utils.config import SystemConfig, GeometryConfig

@dataclass
class ModelConfig:
    """Configuration for base hypercube model."""
    N: int  # Number of units
    J: int  # Number of atoms
    lambda_rate: float  # Arrival rate
    mu_rate: float = 1.0  # Service rate
    district_length: float = 1.0  # Length of each district
    dispatch_policy: str = 'mcm'  # Dispatch policy type

def adapt_config(global_config: GlobalConfig) -> ModelConfig:
    """Adapt global configuration to model configuration.
    
    Args:
        global_config (GlobalConfig): Global configuration object
        
    Returns:
        ModelConfig: Adapted model configuration
    """
    return ModelConfig(
        N=global_config.system.N,
        J=global_config.system.J,
        lambda_rate=global_config.system.lambda_rate,
        mu_rate=global_config.system.mu_rate,
        district_length=global_config.geometry.district_length,
        dispatch_policy=global_config.system.dispatch_type.value
    )