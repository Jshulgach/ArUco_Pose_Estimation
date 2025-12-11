"""
ArUco Pose Estimation Package
A robust toolkit for ArUco marker detection, pose estimation, and custom 3D model tracking.
"""

__version__ = "2.0.0"
__author__ = "Jonathan Shulgach"

from .core.detector import UnifiedArucoDetector
from .core.calibration import CameraCalibrator
from .core.pose_estimator import PoseEstimator, MultiMarkerPoseEstimator
from .utils.visualization import ArucoVisualizer
from .utils.config import ProjectConfig, CameraConfig, CalibrationConfig, ArucoConfig
from .utils.performance import PerformanceMonitor
from .utils.logger import setup_logger

__all__ = [
    'UnifiedArucoDetector',
    'CameraCalibrator',
    'PoseEstimator',
    'MultiMarkerPoseEstimator',
    'ArucoVisualizer',
    'ProjectConfig',
    'CameraConfig',
    'CalibrationConfig',
    'ArucoConfig',
    'PerformanceMonitor',
    'setup_logger',
]
