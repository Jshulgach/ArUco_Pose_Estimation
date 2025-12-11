"""
Professional visualization toolkit for ArUco markers and pose estimation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ArucoVisualizer:
    """
    Professional visualization toolkit for ArUco markers and pose data.
    
    Features:
    - Draw detected markers with customizable styles
    - Draw 3D axes on markers
    - Visualize reprojection errors
    - Display pose information text
    - Color-coded marker IDs
    
    Example:
        >>> viz = ArucoVisualizer(camera_matrix, dist_coeffs)
        >>> image = viz.draw_markers(image, corners, ids, draw_axes=True)
    """
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        Initialize visualizer with camera calibration data.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
        """
        self.K = camera_matrix
        self.D = dist_coeffs
        self.color_map = {}  # Consistent colors per marker ID
        
    def _get_color_for_id(self, marker_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a marker ID"""
        if marker_id not in self.color_map:
            # Generate deterministic color from ID
            np.random.seed(marker_id)
            self.color_map[marker_id] = tuple(np.random.randint(50, 255, 3).tolist())
        return self.color_map[marker_id]
    
    def draw_markers(self, image, corners, ids, 
                     draw_ids=True, 
                     draw_axes=False, 
                     marker_size=0.05, 
                     thickness=2,
                     border_color=(0, 255, 0)):
        """
        Draw detected ArUco markers on image.
        
        Args:
            image: Input image
            corners: Marker corners from detection
            ids: Marker IDs
            draw_ids: Whether to draw marker IDs (default: True)
            draw_axes: Whether to draw 3D axes (default: False)
            marker_size: Physical marker size in meters (for axes)
            thickness: Line thickness
            border_color: Color for marker borders (B,G,R)
            
        Returns:
            Image with drawn markers
        """
        if ids is None or len(corners) == 0:
            return image
        
        image = image.copy()
        
        # Draw marker borders
        cv2.aruco.drawDetectedMarkers(image, corners, ids, borderColor=border_color)
        
        # Draw marker IDs with custom positioning
        if draw_ids:
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]
                # Position above top-left corner
                pos = (int(corner[0][0]), int(corner[0][1] - 10))
                color = self._get_color_for_id(marker_id)
                
                # Draw background rectangle for better visibility
                text = f"ID:{marker_id}"
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(image, 
                            (pos[0]-2, pos[1]-text_height-2),
                            (pos[0]+text_width+2, pos[1]+2),
                            (0, 0, 0), -1)
                
                cv2.putText(image, text, pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw 3D axes if calibration available
        if draw_axes and self.K is not None and self.D is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_size, self.K, self.D
                )
                cv2.drawFrameAxes(image, self.K, self.D, rvec, tvec, 
                                 marker_size/2, thickness)
        
        return image
    
    def draw_pose_info(self, image, rvec, tvec, 
                      position=(10, 30), 
                      font_scale=0.6,
                      color=(0, 255, 0),
                      show_euler=True):
        """
        Draw pose information (position and orientation) on image.
        
        Args:
            image: Input image
            rvec: Rotation vector
            tvec: Translation vector
            position: Text starting position (x, y)
            font_scale: Font size scale
            color: Text color (B,G,R)
            show_euler: Whether to show Euler angles
            
        Returns:
            Image with pose information
        """
        image = image.copy()
        x, y = position
        line_height = int(30 * font_scale)
        
        # Draw translation
        cv2.putText(image, f"X: {tvec[0][0][0]:.4f} m", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        cv2.putText(image, f"Y: {tvec[0][0][1]:.4f} m", 
                   (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        cv2.putText(image, f"Z: {tvec[0][0][2]:.4f} m", 
                   (x, y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        # Draw Euler angles if requested
        if show_euler:
            rot_mat, _ = cv2.Rodrigues(rvec)
            euler_angles = cv2.RQDecomp3x3(rot_mat)[0]
            
            cv2.putText(image, f"Pitch: {euler_angles[0]:.2f} deg", 
                       (x, y + 3*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            cv2.putText(image, f"Roll: {euler_angles[1]:.2f} deg", 
                       (x, y + 4*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            cv2.putText(image, f"Yaw: {euler_angles[2]:.2f} deg", 
                       (x, y + 5*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
        
        return image
    
    def draw_reprojection_error(self, image, obj_pts, img_pts, 
                                rvec, tvec, 
                                error_color=(0, 255, 255),
                                detected_color=(0, 255, 0),
                                projected_color=(0, 0, 255)):
        """
        Visualize reprojection errors for debugging pose estimation.
        
        Draws lines between detected points and their reprojected positions.
        
        Args:
            image: Input image
            obj_pts: 3D object points
            img_pts: Detected 2D image points
            rvec: Rotation vector
            tvec: Translation vector
            error_color: Color for error lines (B,G,R)
            detected_color: Color for detected points (B,G,R)
            projected_color: Color for projected points (B,G,R)
            
        Returns:
            Image with reprojection error visualization
        """
        if self.K is None or self.D is None:
            logger.warning("Camera calibration required for reprojection visualization")
            return image
        
        image = image.copy()
        
        # Project 3D points to 2D
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
        proj_pts = proj_pts.reshape(-1, 2).astype(int)
        img_pts = img_pts.reshape(-1, 2).astype(int)
        
        # Draw error lines and points
        for (x1, y1), (x2, y2) in zip(img_pts, proj_pts):
            cv2.line(image, (x1, y1), (x2, y2), error_color, 1)
            cv2.circle(image, (x1, y1), 4, detected_color, -1)  # Detected
            cv2.circle(image, (x2, y2), 4, projected_color, -1)  # Projected
        
        # Calculate and display RMS error
        errors = np.linalg.norm(img_pts - proj_pts, axis=1)
        rms_error = np.sqrt(np.mean(errors**2))
        
        cv2.putText(image, f"RMS Error: {rms_error:.2f} px", 
                   (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def draw_performance_stats(self, image, stats: dict, 
                              position=(10, 30),
                              font_scale=0.5,
                              color=(255, 255, 0)):
        """
        Draw performance statistics on image.
        
        Args:
            image: Input image
            stats: Dictionary with performance stats (fps, latency, etc.)
            position: Text starting position
            font_scale: Font size scale
            color: Text color (B,G,R)
            
        Returns:
            Image with performance stats
        """
        image = image.copy()
        x, y = position
        line_height = int(25 * font_scale)
        
        for i, (key, value) in enumerate(stats.items()):
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(image, text, (x, y + i*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        return image
    
    def draw_3d_model_overlay(self, image, vertices, faces, 
                             rvec, tvec,
                             color=(255, 255, 0),
                             thickness=2):
        """
        Draw 3D model wireframe overlay on image.
        
        Args:
            image: Input image
            vertices: 3D model vertices (Nx3)
            faces: Face indices (list of lists)
            rvec: Rotation vector
            tvec: Translation vector
            color: Line color (B,G,R)
            thickness: Line thickness
            
        Returns:
            Image with 3D model overlay
        """
        if self.K is None or self.D is None:
            logger.warning("Camera calibration required for 3D overlay")
            return image
        
        image = image.copy()
        
        # Project vertices to 2D
        verts_2d, _ = cv2.projectPoints(vertices, rvec, tvec, self.K, self.D)
        verts_2d = verts_2d.reshape(-1, 2).astype(int)
        
        # Draw faces
        for face in faces:
            pts = verts_2d[face]
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
        
        return image
