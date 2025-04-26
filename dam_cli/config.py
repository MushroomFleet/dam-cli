"""
Configuration handling for the DAM CLI Wrapper
"""
import json
import os
from argparse import Namespace
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")


def merge_config_with_args(config: Dict[str, Any], args: Namespace) -> Namespace:
    """
    Merge configuration from file with command line arguments.
    Command line arguments take precedence over configuration file.
    
    Args:
        config: Configuration dictionary from file
        args: Command line arguments
        
    Returns:
        Updated arguments with merged configuration
    """
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Update with config, but don't overwrite args that were explicitly set
    for key, value in config.items():
        # Skip if the argument was explicitly provided on the command line
        # This assumes the default value is None for most options
        if key in args_dict and args_dict[key] is not None:
            continue
        
        # For boolean flags that default to False, check if they were explicitly set
        if key in args_dict and isinstance(args_dict[key], bool):
            # Skip boolean flags that were set on the command line
            param_name = f"--{key}"
            is_default = True  # Without parsing the argparse source, we assume default
            if not is_default:
                continue
        
        # Update argument with config value
        setattr(args, key, value)
    
    return args


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def args_to_config(args: Namespace) -> Dict[str, Any]:
    """
    Convert command line arguments to configuration dictionary.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Copy non-None values
    for key, value in args_dict.items():
        if value is not None:
            config[key] = value
    
    return config
