"""
3D Models and Pose Estimation classes.

This module contains:
- Dodecahedron and custom 3D model geometry
- Multi-marker pose estimation and fusion
- ArUco marker-based pose tracking
"""

from .dodecahedron_model import CleanDodecahedronModel
from .aruco_pose_pipeline import ArucoPoseEstimator
from .aruco_pose_estimator import ArucoPoseEstimatorFromJSON

__all__ = [
    'CleanDodecahedronModel',
    'ArucoPoseEstimator',
    'ArucoPoseEstimatorFromJSON',
]
