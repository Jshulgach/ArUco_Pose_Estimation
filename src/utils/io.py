"""
Video I/O and data recording utilities.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class VideoHandler:
    """
    Handle various video sources with robust error handling.
    
    Supports:
    - Webcam (device ID)
    - Video files
    - IP cameras (HTTP/RTSP streams)
    - Image sequences
    
    Example:
        >>> handler = VideoHandler(source=0)  # Webcam
        >>> ret, frame = handler.read()
        >>> handler.release()
    """
    
    def __init__(self, source: Union[int, str], 
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 fps: Optional[int] = None):
        """
        Initialize video handler.
        
        Args:
            source: Camera ID (int), file path (str), or stream URL (str)
            width: Desired frame width (optional)
            height: Desired frame height (optional)
            fps: Desired frame rate (optional)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If unable to open video source
        """
        self.source = source
        self.cap = None
        self.is_camera = isinstance(source, int)
        self.is_stream = isinstance(source, str) and source.startswith(('http://', 'rtsp://'))
        
        self._initialize(width, height, fps)
        
        logger.info(f"Initialized video source: {source}")
    
    def _initialize(self, width: Optional[int], height: Optional[int], fps: Optional[int]):
        """Initialize video capture"""
        # Check if file exists
        if isinstance(self.source, str) and not self.is_stream:
            if not Path(self.source).exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")
        
        # Open video capture
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        # Set properties if specified
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None and self.is_camera:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def read(self):
        """
        Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        return self.cap.read()
    
    def get_properties(self):
        """
        Get video properties.
        
        Returns:
            Dictionary with video properties
        """
        if self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
        }
    
    def release(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Released video source")
    
    def is_opened(self):
        """Check if video source is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


class PoseRecorder:
    """
    Record pose data and marker detections for analysis.
    
    Records:
    - Timestamps
    - Rotation and translation vectors
    - Marker IDs
    - Reprojection errors
    - Optional metadata
    
    Example:
        >>> recorder = PoseRecorder("output/poses.json")
        >>> recorder.record_frame(time.time(), rvec, tvec, [0, 1, 2])
        >>> recorder.save()
    """
    
    def __init__(self, output_path: str, save_interval: int = 100):
        """
        Initialize pose recorder.
        
        Args:
            output_path: Path to output JSON file
            save_interval: Auto-save every N frames (0 to disable)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.data = []
        self.save_interval = save_interval
        self.frame_count = 0
        
        logger.info(f"Initialized pose recorder: {output_path}")
    
    def record_frame(self, timestamp: float, 
                    rvec: Optional[np.ndarray] = None,
                    tvec: Optional[np.ndarray] = None,
                    marker_ids: Optional[np.ndarray] = None,
                    reprojection_error: Optional[float] = None,
                    metadata: Optional[dict] = None):
        """
        Record a frame of pose data.
        
        Args:
            timestamp: Frame timestamp (seconds)
            rvec: Rotation vector
            tvec: Translation vector
            marker_ids: Detected marker IDs
            reprojection_error: RMS reprojection error
            metadata: Additional metadata dictionary
        """
        frame_data = {
            'frame': self.frame_count,
            'timestamp': timestamp,
            'rvec': rvec.flatten().tolist() if rvec is not None else None,
            'tvec': tvec.flatten().tolist() if tvec is not None else None,
            'marker_ids': marker_ids.tolist() if marker_ids is not None else [],
            'reprojection_error': reprojection_error,
        }
        
        if metadata:
            frame_data['metadata'] = metadata
        
        self.data.append(frame_data)
        self.frame_count += 1
        
        # Auto-save if interval reached
        if self.save_interval > 0 and self.frame_count % self.save_interval == 0:
            self.save()
            logger.debug(f"Auto-saved at frame {self.frame_count}")
    
    def save(self):
        """Save recorded data to JSON file"""
        with open(self.output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'total_frames': self.frame_count,
                    'format_version': '1.0'
                },
                'frames': self.data
            }, f, indent=2)
        
        logger.info(f"Saved {self.frame_count} frames to {self.output_path}")
    
    def load(self, path: str):
        """
        Load recorded data from JSON file.
        
        Args:
            path: Path to JSON file
        """
        with open(path, 'r') as f:
            loaded = json.load(f)
        
        self.data = loaded.get('frames', [])
        self.frame_count = len(self.data)
        
        logger.info(f"Loaded {self.frame_count} frames from {path}")
    
    def get_statistics(self):
        """
        Get statistics about recorded data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.data:
            return {}
        
        successful_poses = sum(1 for d in self.data if d['rvec'] is not None)
        total_markers = sum(len(d['marker_ids']) for d in self.data)
        
        errors = [d['reprojection_error'] for d in self.data 
                 if d['reprojection_error'] is not None]
        
        stats = {
            'total_frames': self.frame_count,
            'successful_poses': successful_poses,
            'success_rate': successful_poses / self.frame_count if self.frame_count > 0 else 0,
            'total_markers_detected': total_markers,
            'avg_markers_per_frame': total_markers / self.frame_count if self.frame_count > 0 else 0,
        }
        
        if errors:
            stats['avg_reprojection_error'] = np.mean(errors)
            stats['max_reprojection_error'] = np.max(errors)
            stats['min_reprojection_error'] = np.min(errors)
        
        return stats
    
    def export_to_csv(self, output_path: str):
        """
        Export data to CSV format.
        
        Args:
            output_path: Path to output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['frame', 'timestamp', 'rx', 'ry', 'rz', 
                           'tx', 'ty', 'tz', 'marker_ids', 'reproj_error'])
            
            # Data
            for frame_data in self.data:
                rvec = frame_data['rvec'] if frame_data['rvec'] else [None, None, None]
                tvec = frame_data['tvec'] if frame_data['tvec'] else [None, None, None]
                
                writer.writerow([
                    frame_data['frame'],
                    frame_data['timestamp'],
                    rvec[0] if rvec[0] is not None else '',
                    rvec[1] if rvec[1] is not None else '',
                    rvec[2] if rvec[2] is not None else '',
                    tvec[0] if tvec[0] is not None else '',
                    tvec[1] if tvec[1] is not None else '',
                    tvec[2] if tvec[2] is not None else '',
                    ','.join(map(str, frame_data['marker_ids'])),
                    frame_data['reprojection_error'] if frame_data['reprojection_error'] else ''
                ])
        
        logger.info(f"Exported to CSV: {output_path}")
