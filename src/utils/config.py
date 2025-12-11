"""
Configuration management with validation using dataclasses.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    device_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: Optional[int] = None
    
    def validate(self):
        """Validate camera configuration"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("FPS must be positive")


@dataclass
class CalibrationConfig:
    """Calibration configuration parameters"""
    checkerboard_rows: int = 5
    checkerboard_cols: int = 7
    square_size: float = 0.0319  # meters
    n_frames: int = 10
    img_dir: str = "frames"
    cooldown: int = 100
    view_resize: float = 2.0
    
    def validate(self):
        """Validate calibration configuration"""
        if self.checkerboard_rows <= 0 or self.checkerboard_cols <= 0:
            raise ValueError("Checkerboard dimensions must be positive")
        if self.square_size <= 0:
            raise ValueError("Square size must be positive")
        if self.n_frames <= 0:
            raise ValueError("Number of frames must be positive")
        if self.view_resize <= 0:
            raise ValueError("View resize factor must be positive")


@dataclass
class ArucoConfig:
    """ArUco marker configuration parameters"""
    dict_type: str = "DICT_5X5_100"
    marker_size: float = 0.015  # meters
    
    def validate(self):
        """Validate ArUco configuration"""
        valid_dicts = [
            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
            "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
            "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
            "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"
        ]
        if self.dict_type not in valid_dicts:
            raise ValueError(f"Invalid ArUco dictionary type: {self.dict_type}")
        if self.marker_size <= 0:
            raise ValueError("Marker size must be positive")


@dataclass
class ProjectConfig:
    """Main project configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    
    @classmethod
    def from_yaml(cls, path: str):
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            ProjectConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        camera_data = data.get('camera', {})
        calibration_data = data.get('calibration', {})
        aruco_data = data.get('aruco', {})
        
        config = cls(
            camera=CameraConfig(**camera_data),
            calibration=CalibrationConfig(**calibration_data),
            aruco=ArucoConfig(**aruco_data)
        )
        
        logger.info(f"Loaded configuration from {path}")
        return config
    
    @classmethod
    def from_legacy_yaml(cls, path: str):
        """
        Load from legacy config.yaml format for backward compatibility.
        
        Args:
            path: Path to legacy YAML file
            
        Returns:
            ProjectConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Map legacy keys to new structure
        camera_data = {
            'device_id': data.get('camera0', 0),
            'width': data.get('frame_width', 1920),
            'height': data.get('frame_height', 1080),
        }
        
        calibration_data = {
            'checkerboard_rows': data.get('checkerboard_row_vertices', 5),
            'checkerboard_cols': data.get('checkerboard_column_vertices', 7),
            'square_size': data.get('checkerboard_box_size_scale', 0.0319),
            'n_frames': data.get('n_frames', 10),
            'img_dir': data.get('img_dir', 'frames'),
            'cooldown': data.get('cooldown', 100),
            'view_resize': data.get('view_resize', 2.0),
        }
        
        aruco_data = {
            'marker_size': data.get('aruco_size', 0.015),
        }
        
        config = cls(
            camera=CameraConfig(**camera_data),
            calibration=CalibrationConfig(**calibration_data),
            aruco=ArucoConfig(**aruco_data)
        )
        
        logger.info(f"Loaded legacy configuration from {path}")
        return config
    
    def validate(self):
        """Validate all configuration sections"""
        self.camera.validate()
        self.calibration.validate()
        self.aruco.validate()
    
    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to output YAML file
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'camera': asdict(self.camera),
            'calibration': asdict(self.calibration),
            'aruco': asdict(self.aruco)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'camera': asdict(self.camera),
            'calibration': asdict(self.calibration),
            'aruco': asdict(self.aruco)
        }
