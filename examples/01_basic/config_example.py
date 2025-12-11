"""
Configuration management example.

This example shows how to:
1. Create configurations programmatically
2. Load from YAML files
3. Validate configurations
4. Save configurations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ProjectConfig, CameraConfig, CalibrationConfig, ArucoConfig
from src.utils.logger import setup_logger

logger = setup_logger("config_example", level=20)


def example_create_config():
    """Create configuration programmatically"""
    logger.info("=== Creating configuration programmatically ===")
    
    # Create with defaults
    config = ProjectConfig()
    
    # Or create with custom values
    config = ProjectConfig(
        camera=CameraConfig(
            device_id=0,
            width=1920,
            height=1080,
            fps=30
        ),
        calibration=CalibrationConfig(
            checkerboard_rows=5,
            checkerboard_cols=7,
            square_size=0.025,
            n_frames=15
        ),
        aruco=ArucoConfig(
            dict_type="DICT_5X5_100",
            marker_size=0.05
        )
    )
    
    # Validate
    try:
        config.validate()
        logger.info("✓ Configuration is valid")
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
    
    # Save to YAML
    config.to_yaml("example_config.yaml")
    logger.info("Saved configuration to example_config.yaml")
    
    return config


def example_load_config():
    """Load configuration from YAML"""
    logger.info("\n=== Loading configuration from YAML ===")
    
    try:
        # Load from new format
        config = ProjectConfig.from_yaml("example_config.yaml")
        logger.info("✓ Loaded configuration")
        
        # Access config values
        logger.info(f"Camera: {config.camera.width}x{config.camera.height}")
        logger.info(f"Marker size: {config.aruco.marker_size} m")
        
        return config
    
    except FileNotFoundError:
        logger.warning("Config file not found")
        return None


def example_legacy_config():
    """Load legacy config format"""
    logger.info("\n=== Loading legacy configuration ===")
    
    try:
        # Load from legacy config.yaml
        config = ProjectConfig.from_legacy_yaml("../config.yaml")
        logger.info("✓ Loaded legacy configuration")
        
        # Convert to new format
        config.to_yaml("converted_config.yaml")
        logger.info("Saved as new format: converted_config.yaml")
        
        return config
    
    except FileNotFoundError:
        logger.warning("Legacy config file not found")
        return None


def main():
    # Create and save config
    config = example_create_config()
    
    # Load config
    loaded_config = example_load_config()
    
    # Try loading legacy config
    legacy_config = example_legacy_config()
    
    # Convert to dictionary
    if config:
        config_dict = config.to_dict()
        logger.info(f"\n=== Configuration as dictionary ===")
        logger.info(f"{config_dict}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
