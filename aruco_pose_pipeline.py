import cv2
import numpy as np
import json
from pathlib import Path
from packaging.version import Version


class ArucoPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, aruco_dict=cv2.aruco.DICT_5X5_100, tag_data_path="aruco_group.json"):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_data = self._load_tag_data(tag_data_path)

        # Support for multiple cv2 versions
        if Version(cv2.__version__) >= Version("4.7.0"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None


    def _load_tag_data(self, json_path):
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Could not find {json_path} for ArUco extrinsics.")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data["tags"]

    def get_tag_extrinsics(self, tag_id):
        tag = self.tag_data[str(tag_id)]
        tvec = np.array(tag["extrinsics"][:3], dtype=np.float32).reshape(3, 1)
        rvec = np.array(tag["extrinsics"][3:], dtype=np.float32).reshape(3, 1)
        return rvec, tvec

    def get_tag_object_points(self, size):
        half = size / 2.0
        return np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

    def detect_and_estimate_pose(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(image=gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        rvec, tvec = None, None

        if ids is not None and len(ids) >= 1:

            ids = ids.flatten()
            if 0 in ids:
                i = np.where(ids == 0)[0][0]
                size = self.tag_data["0"]["size"]
                obj_pts = self.get_tag_object_points(size)
                r_tag, t_tag = self.get_tag_extrinsics(0)
                transformed = self.transform_marker_corners(obj_pts, (r_tag, t_tag))
                image_corners = corners[i][0]

                success, rvec, tvec = cv2.solvePnP(np.array(transformed, dtype=np.float32),
                                                   np.array(image_corners, dtype=np.float32), self.camera_matrix,
                                                   self.dist_coeffs)
                if success:
                    return rvec, tvec, corners, ids

            # all_obj_pts = []
            # all_img_pts = []
            # for i, marker_id in enumerate(ids.flatten()):
            #     if str(marker_id) not in self.tag_data:
            #         continue
            #     size = self.tag_data[str(marker_id)]["size"]
            #     obj_pts = self.get_tag_object_points(size)
            #     r_tag, t_tag = self.get_tag_extrinsics(marker_id)
            #     transformed = self.transform_marker_corners(obj_pts, (r_tag, t_tag))
            #     all_obj_pts.extend(transformed)
            #     all_img_pts.extend(corners[i][0])
            #
            # if len(all_obj_pts) >= 4:
            #     all_obj_pts = np.array(all_obj_pts, dtype=np.float32)
            #     all_img_pts = np.array(all_img_pts, dtype=np.float32)
            #     success, rvec, tvec = cv2.solvePnP(all_obj_pts, all_img_pts, self.camera_matrix, self.dist_coeffs)
            #     if success:
            #         return rvec, tvec, corners, ids

        return None, None, corners, ids

    def transform_marker_corners(self, obj_pts, transformation):
        rmat = cv2.Rodrigues(transformation[0])[0]
        rotated = obj_pts @ rmat.T
        translated = rotated + transformation[1].T
        return translated.reshape(-1, 3)

    def draw_debug_overlay(self, image, rvec, tvec):
        axis = np.float32([[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        corner = tuple(imgpts[2].ravel().astype(int))
        image = cv2.line(image, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3)
        image = cv2.line(image, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)
        image = cv2.line(image, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3)
        return image
