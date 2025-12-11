"""
Performance monitoring for detection and pose estimation.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Track detection and pose estimation performance metrics.
    
    Monitors:
    - Detection FPS
    - Pose estimation FPS
    - Reprojection errors
    - Processing latency
    - Marker detection rate
    
    Example:
        >>> monitor = PerformanceMonitor(window_size=30)
        >>> monitor.start_frame()
        >>> # ... do detection ...
        >>> monitor.record_detection(num_markers=5)
        >>> # ... do pose estimation ...
        >>> monitor.end_frame()
        >>> stats = monitor.get_stats()
    """
    
    def __init__(self, window_size=30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.pose_times = deque(maxlen=window_size)
        self.reprojection_errors = deque(maxlen=window_size)
        self.markers_detected = deque(maxlen=window_size)
        
        self._frame_start_time = None
        self._detection_start_time = None
        self._pose_start_time = None
        
        self.total_frames = 0
        self.total_markers = 0
    
    def start_frame(self):
        """Mark the start of a frame"""
        self._frame_start_time = time.perf_counter()
    
    def end_frame(self):
        """Mark the end of a frame"""
        if self._frame_start_time is not None:
            duration = time.perf_counter() - self._frame_start_time
            self.frame_times.append(duration)
            self.total_frames += 1
            self._frame_start_time = None
    
    def start_detection(self):
        """Mark the start of detection"""
        self._detection_start_time = time.perf_counter()
    
    def end_detection(self, num_markers: int = 0):
        """
        Mark the end of detection.
        
        Args:
            num_markers: Number of markers detected
        """
        if self._detection_start_time is not None:
            duration = time.perf_counter() - self._detection_start_time
            self.detection_times.append(duration)
            self.markers_detected.append(num_markers)
            self.total_markers += num_markers
            self._detection_start_time = None
    
    def record_detection(self, num_markers: int, duration: Optional[float] = None):
        """
        Record detection metrics (alternative to start/end).
        
        Args:
            num_markers: Number of markers detected
            duration: Detection time in seconds (optional)
        """
        self.markers_detected.append(num_markers)
        self.total_markers += num_markers
        if duration is not None:
            self.detection_times.append(duration)
    
    def start_pose_estimation(self):
        """Mark the start of pose estimation"""
        self._pose_start_time = time.perf_counter()
    
    def end_pose_estimation(self):
        """Mark the end of pose estimation"""
        if self._pose_start_time is not None:
            duration = time.perf_counter() - self._pose_start_time
            self.pose_times.append(duration)
            self._pose_start_time = None
    
    def record_pose_time(self, duration: float):
        """
        Record pose estimation time.
        
        Args:
            duration: Pose estimation time in seconds
        """
        self.pose_times.append(duration)
    
    def add_reprojection_error(self, error: float):
        """
        Add reprojection error measurement.
        
        Args:
            error: RMS reprojection error in pixels
        """
        self.reprojection_errors.append(error)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {}
        
        # Frame rate
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            stats['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            stats['frame_time_ms'] = avg_frame_time * 1000
        else:
            stats['fps'] = 0
            stats['frame_time_ms'] = 0
        
        # Detection metrics
        if self.detection_times:
            avg_detection_time = np.mean(self.detection_times)
            stats['detection_fps'] = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
            stats['detection_time_ms'] = avg_detection_time * 1000
        else:
            stats['detection_fps'] = 0
            stats['detection_time_ms'] = 0
        
        # Pose estimation metrics
        if self.pose_times:
            avg_pose_time = np.mean(self.pose_times)
            stats['pose_fps'] = 1.0 / avg_pose_time if avg_pose_time > 0 else 0
            stats['pose_time_ms'] = avg_pose_time * 1000
        else:
            stats['pose_fps'] = 0
            stats['pose_time_ms'] = 0
        
        # Marker detection rate
        if self.markers_detected:
            stats['avg_markers'] = np.mean(self.markers_detected)
            stats['detection_rate'] = sum(1 for n in self.markers_detected if n > 0) / len(self.markers_detected)
        else:
            stats['avg_markers'] = 0
            stats['detection_rate'] = 0
        
        # Reprojection error
        if self.reprojection_errors:
            stats['avg_reproj_error'] = np.mean(self.reprojection_errors)
            stats['max_reproj_error'] = np.max(self.reprojection_errors)
        else:
            stats['avg_reproj_error'] = 0
            stats['max_reproj_error'] = 0
        
        # Totals
        stats['total_frames'] = self.total_frames
        stats['total_markers'] = self.total_markers
        
        return stats
    
    def get_summary(self) -> str:
        """
        Get formatted summary string.
        
        Returns:
            Formatted statistics string
        """
        stats = self.get_stats()
        
        summary = [
            "=== Performance Summary ===",
            f"Overall FPS: {stats['fps']:.1f}",
            f"Frame Time: {stats['frame_time_ms']:.1f} ms",
            f"Detection Time: {stats['detection_time_ms']:.1f} ms",
            f"Pose Time: {stats['pose_time_ms']:.1f} ms",
            f"Avg Markers: {stats['avg_markers']:.1f}",
            f"Detection Rate: {stats['detection_rate']*100:.1f}%",
            f"Total Frames: {stats['total_frames']}",
            f"Total Markers: {stats['total_markers']}",
        ]
        
        if stats['avg_reproj_error'] > 0:
            summary.extend([
                f"Avg Reproj Error: {stats['avg_reproj_error']:.2f} px",
                f"Max Reproj Error: {stats['max_reproj_error']:.2f} px",
            ])
        
        return "\n".join(summary)
    
    def reset(self):
        """Reset all statistics"""
        self.frame_times.clear()
        self.detection_times.clear()
        self.pose_times.clear()
        self.reprojection_errors.clear()
        self.markers_detected.clear()
        self.total_frames = 0
        self.total_markers = 0
        logger.info("Performance monitor reset")
    
    def log_stats(self):
        """Log current statistics"""
        logger.info(f"\n{self.get_summary()}")
