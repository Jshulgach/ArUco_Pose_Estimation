"""
Pose estimation example with performance monitoring.

This example shows how to:
1. Load camera calibration
2. Estimate pose from ArUco markers
3. Monitor performance
4. Visualize results with 3D axes
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import UnifiedArucoDetector
from src.utils.visualization import ArucoVisualizer
from src.utils.performance import PerformanceMonitor
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("pose_estimation", level=20)


def main():
    # Load calibration (you need to run calibration first!)
    try:
        K = np.load("camera_intrinsics.npy")
        D = np.load("camera_distortion.npy")
        logger.info("Loaded camera calibration")
    except FileNotFoundError:
        logger.error("Calibration files not found. Please run calibration first!")
        return 1
    
    # Initialize components
    detector = UnifiedArucoDetector("DICT_5X5_100")
    visualizer = ArucoVisualizer(K, D)
    monitor = PerformanceMonitor()
    
    # Marker size in meters
    MARKER_SIZE = 0.05
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return 1
    
    logger.info("Starting pose estimation... (Press 'q' to quit, 'r' to reset stats)")
    
    try:
        while True:
            monitor.start_frame()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect markers
            corners, ids, rejected = detector.detect(frame)
            n_markers = len(ids) if ids is not None else 0
            monitor.record_detection(n_markers)
            
            # Draw markers with 3D axes
            frame = visualizer.draw_markers(frame, corners, ids, 
                                           draw_axes=True, 
                                           marker_size=MARKER_SIZE)
            
            # Estimate pose for first marker
            if ids is not None and len(ids) > 0:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[0], MARKER_SIZE, K, D
                )
                
                # Draw pose information
                frame = visualizer.draw_pose_info(frame, rvec, tvec)
            
            # Show performance stats
            stats = monitor.get_stats()
            display_stats = {
                'FPS': stats['fps'],
                'Detection Time': f"{stats['detection_time_ms']:.1f} ms",
                'Markers': n_markers
            }
            frame = visualizer.draw_performance_stats(
                frame, display_stats, 
                position=(frame.shape[1]-250, 30)
            )
            
            monitor.end_frame()
            
            # Display
            cv2.imshow('ArUco Pose Estimation', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                monitor.reset()
                logger.info("Reset statistics")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        logger.info("\n" + monitor.get_summary())
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
