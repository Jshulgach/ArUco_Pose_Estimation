"""
Unified ArUco detector that handles different OpenCV versions seamlessly.
"""

import cv2
import numpy as np
from packaging.version import Version
import logging

logger = logging.getLogger(__name__)

# Dictionary mapping of ArUco names to OpenCV dictionary IDs
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


class UnifiedArucoDetector:
    """
    Handles ArUco marker detection across different OpenCV versions.
    
    Automatically detects OpenCV version and uses appropriate API.
    Supports both legacy (< 4.7.0) and modern OpenCV versions.
    
    Args:
        aruco_dict_type: ArUco dictionary type (string or OpenCV constant)
        params: Detection parameters (optional)
        
    Example:
        >>> detector = UnifiedArucoDetector("DICT_5X5_100")
        >>> corners, ids, rejected = detector.detect(image)
    """
    
    def __init__(self, aruco_dict_type="DICT_5X5_100", params=None):
        """
        Initialize the detector with specified dictionary type.
        
        Args:
            aruco_dict_type: String name or OpenCV constant for ArUco dictionary
            params: Optional DetectorParameters for customization
        """
        if isinstance(aruco_dict_type, str):
            if aruco_dict_type not in ARUCO_DICT:
                raise ValueError(
                    f"Invalid ArUco dictionary type: {aruco_dict_type}. "
                    f"Must be one of {list(ARUCO_DICT.keys())}"
                )
            self.dict_type = ARUCO_DICT[aruco_dict_type]
            self.dict_name = aruco_dict_type
        else:
            self.dict_type = aruco_dict_type
            self.dict_name = "custom"
        
        self._setup_detector(params)
        logger.info(f"Initialized ArUco detector with {self.dict_name} "
                   f"(OpenCV {cv2.__version__}, legacy={self.use_legacy})")
    
    def _setup_detector(self, params):
        """
        Auto-detect OpenCV version and setup appropriate detector.
        
        Args:
            params: Optional detection parameters
        """
        cv_version = Version(cv2.__version__)
        
        if cv_version >= Version("4.7.0"):
            # Modern OpenCV API (>= 4.7.0)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dict_type)
            self.params = params if params is not None else cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
            self.use_legacy = False
        else:
            # Legacy OpenCV API (< 4.7.0)
            self.aruco_dict = cv2.aruco.Dictionary_get(self.dict_type)
            self.params = params if params is not None else cv2.aruco.DetectorParameters_create()
            self.detector = None
            self.use_legacy = True
    
    def detect(self, image, convert_to_gray=True):
        """
        Detect ArUco markers in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            convert_to_gray: Whether to convert to grayscale (default: True)
            
        Returns:
            Tuple of (corners, ids, rejected_candidates)
            - corners: List of detected marker corners
            - ids: Array of marker IDs
            - rejected_candidates: List of rejected marker candidates
            
        Raises:
            ValueError: If image is None or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        # Convert to grayscale if needed
        if convert_to_gray and len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect markers using appropriate API
        if self.use_legacy:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.params
            )
        else:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Log detection results
        n_detected = len(ids) if ids is not None else 0
        logger.debug(f"Detected {n_detected} markers, {len(rejected)} rejected")
        
        return corners, ids, rejected
    
    def get_marker_size(self, corners):
        """
        Calculate the average size of detected markers in pixels.
        
        Args:
            corners: Marker corners from detection
            
        Returns:
            Average marker size in pixels
        """
        if not corners:
            return 0.0
        
        sizes = []
        for corner in corners:
            pts = corner[0]
            # Calculate average edge length
            edge_lengths = [
                np.linalg.norm(pts[i] - pts[(i+1) % 4])
                for i in range(4)
            ]
            sizes.append(np.mean(edge_lengths))
        
        return np.mean(sizes)
    
    def update_parameters(self, **kwargs):
        """
        Update detector parameters dynamically.
        
        Args:
            **kwargs: Parameter names and values to update
            
        Example:
            >>> detector.update_parameters(
            ...     adaptiveThreshConstant=7,
            ...     minMarkerPerimeterRate=0.03
            ... )
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
                logger.debug(f"Updated parameter {key} = {value}")
            else:
                logger.warning(f"Parameter {key} not found in DetectorParameters")
        
        # Recreate detector with updated parameters if using modern API
        if not self.use_legacy:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
    
    def __repr__(self):
        return (f"UnifiedArucoDetector(dict={self.dict_name}, "
                f"opencv_version={cv2.__version__}, legacy={self.use_legacy})")
