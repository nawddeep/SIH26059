"""Configuration management for sea-ice forecasting model."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default settings.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "settings.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Config with resolved paths
    """
    root = get_project_root()
    
    # Resolve data paths
    if 'data' in config and 'paths' in config['data']:
        for key, path in config['data']['paths'].items():
            config['data']['paths'][key] = str(root / path)
    
    # Resolve output paths
    if 'output' in config:
        for key in ['model_dir', 'results_dir', 'plots_dir', 'logs_dir']:
            if key in config['output']:
                config['output'][key] = str(root / config['output'][key])
    
    return config
