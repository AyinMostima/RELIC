"""
Configuration I/O utilities for isotope tracking module.

This module provides functions for saving and loading isotope configuration
files in YAML and JSON formats.
"""

import json
import yaml
from pathlib import Path
from typing import Union
from .multi_isotope_models import IsotopeConfig


def save_isotope_config_to_file(config: IsotopeConfig, file_path: Union[str, Path]) -> None:
    """
    Save isotope configuration to file.
    
    Parameters:
        config: IsotopeConfig object to save
        file_path: Path to output file (.yaml, .yml, or .json)
        
    Raises:
        ValueError: If file format is not supported
        IOError: If file cannot be written
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get configuration as dictionary
    config_dict = config.model_dump()
    
    if file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif file_path.suffix == '.json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_isotope_config_from_file(file_path: Union[str, Path]) -> IsotopeConfig:
    """
    Load isotope configuration from file.
    
    Parameters:
        file_path: Path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        IsotopeConfig object loaded from file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or content is invalid
        yaml.YAMLError: If YAML parsing fails
        json.JSONDecodeError: If JSON parsing fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            config_dict = yaml.safe_load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Create IsotopeConfig from dictionary
    return IsotopeConfig(**config_dict)
