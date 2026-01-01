#!/usr/bin/env python
"""
Configuration loader for SHARP v3.2.x training.
Loads configuration from INI file and creates a `SHARPTrainingConfig`.
"""

import argparse
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from sharp_training_script_fixed import SHARPTrainingConfig, DedicatedSHARPTrainer
except ImportError:  # pragma: no cover - legacy fallback
    from sharp_training_script import (  # type: ignore
        SHARPTrainingConfig,
        DedicatedSHARPTrainer,
    )


# Type alias for configuration values - more specific than Any
ConfigValue = Union[bool, int, float, str, None]


def parse_config_value(value: str) -> ConfigValue:
    """Parse configuration value to appropriate type.

    Returns:
        Parsed value as bool, int, float, str, or None
    """
    # Boolean values
    if value.lower() in ['true', 'yes', 'on']:
        return True
    elif value.lower() in ['false', 'no', 'off']:
        return False
    
    # Empty string
    if value.strip() == '':
        return None
    
    # Try to parse as number
    try:
        # Try integer first
        if '.' not in value:
            return int(value)
        else:
            return float(value)
    except ValueError:
        # Return as string
        return value


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from INI file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Flatten configuration into single dictionary
    config_dict = {}
    for section in config.sections():
        for key, value in config[section].items():
            # Handle special cases
            if key == 'val_crop_height':
                if 'val_crop_size' not in config_dict:
                    config_dict['val_crop_size'] = [None, None]
                config_dict['val_crop_size'][0] = parse_config_value(value)
            elif key == 'val_crop_width':
                if 'val_crop_size' not in config_dict:
                    config_dict['val_crop_size'] = [None, None]
                config_dict['val_crop_size'][1] = parse_config_value(value)
            else:
                config_dict[key] = parse_config_value(value)
    
    # Convert val_crop_size to tuple if present
    if 'val_crop_size' in config_dict:
        config_dict['val_crop_size'] = tuple(config_dict['val_crop_size'])
    
    return config_dict


def create_training_config(config_dict: Dict[str, Any], 
                          overrides: Optional[Dict[str, Any]] = None) -> SHARPTrainingConfig:
    """Create SHARPTrainingConfig from configuration dictionary"""
    # Apply any command-line overrides
    if overrides:
        config_dict.update(overrides)
    
    # Filter out None values
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    
    # Create config object
    return SHARPTrainingConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description='Load SHARP v3.2.2 configuration and start training'
    )
    
    parser.add_argument(
        'config', 
        type=str,
        help='Path to configuration file'
    )
    
    # Allow command-line overrides for key parameters
    parser.add_argument('--model_size', type=str, help='Override model size')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, dest='learning_rate', help='Override learning rate')
    parser.add_argument('--sparsity', type=float, dest='sparse_sparsity_ratio', 
                       help='Override sparsity ratio')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--experiment_name', type=str, help='Override experiment name')
    parser.add_argument('--resume', type=str, dest='resume_from', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration file
    config_path = Path(args.config)
    print(f"Loading configuration from: {config_path}")
    config_dict = load_config_file(config_path)
    
    # Prepare overrides
    overrides = {}
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            overrides[key] = value
    
    if overrides:
        print(f"Applying command-line overrides: {overrides}")
    
    # Create training config
    training_config = create_training_config(config_dict, overrides)
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Model: SHARP {training_config.model_size}")
    print(f"  Sparsity: {training_config.sparse_sparsity_ratio}")
    print(f"  RBF centers: {training_config.rbf_centers_per_head}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Output: {training_config.output_dir}/{training_config.experiment_name or 'auto'}")
    
    # Create trainer and start training
    trainer = DedicatedSHARPTrainer(training_config)
    trainer.train()


if __name__ == '__main__':
    main()
