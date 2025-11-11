
import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config dictionary
    """
    path = Path(config_path)
    
    if path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to file."""
    path = Path(output_path)
    
    if path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
