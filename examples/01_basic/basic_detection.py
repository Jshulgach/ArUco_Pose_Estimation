"""
Basic ArUco marker detection example.

This example shows how to:
1. Initialize the detector
2. Capture video from webcam
3. Detect ArUco markers
4. Display results
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import UnifiedArucoDetector
from src.utils.visualization import ArucoVisualizer
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("basic_detection", level=20)


def main():
    # Initialize detector
    detector = UnifiedArucoDetector("DICT_5X5_100")
    visualizer = ArucoVisualizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return 1
    
    logger.info("Starting detection... (Press 'q' to quit)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect markers
            corners, ids, rejected = detector.detect(frame)
            
            # Draw markers
            frame = visualizer.draw_markers(frame, corners, ids)
            
            # Display
            cv2.imshow('Basic ArUco Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    logger.info("Detection stopped")
    return 0


if __name__ == '__main__':
    sys.exit(main())
