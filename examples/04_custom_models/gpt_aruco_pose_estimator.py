import cv2
import numpy as np
import json

class ArucoPoseEstimatorFromJSON:
    def __init__(self, k_matrix, d_coeffs, marker_size, json_path, aruco_dict=cv2.aruco.DICT_5X5_100):
        self.k = k_matrix
        self.d = d_coeffs
        self.marker_size = marker_size
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        with open(json_path, "r") as f:
            tag_data = json.load(f)["faces"]

        self.extrinsics = {}
        for entry in tag_data:
            aruco_id = entry["aruco_id"]
            extr = entry["extrinsics"]
            tvec = np.array(extr[:3], dtype=np.float32)
            rvec = np.array(extr[3:], dtype=np.float32)
            self.extrinsics[aruco_id] = (rvec, tvec)

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        return corners, ids

    def estimate_pose(self, corners, ids):
        obj_pts = []
        img_pts = []
        used_ids = []

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.extrinsics:
                    rvec, tvec = self.extrinsics[marker_id]
                    size = self.marker_size / 2
                    obj = np.array([
                        [-size,  size, 0],
                        [ size,  size, 0],
                        [ size, -size, 0],
                        [-size, -size, 0]
                    ], dtype=np.float32)
                    R, _ = cv2.Rodrigues(rvec)
                    obj_world = (R @ obj.T).T + tvec

                    obj_pts.extend(obj_world)
                    img_pts.extend(corners[i][0])
                    used_ids.append(marker_id)

        if len(obj_pts) >= 4:
            obj_pts = np.array(obj_pts, dtype=np.float32)
            img_pts = np.array(img_pts, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.k, self.d, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                return rvec, tvec, used_ids
        return None, None, used_ids

class ArucoPoseEstimatorFromModel:
    def __init__(self, k_matrix, d_coeffs, marker_size, model, aruco_dict=cv2.aruco.DICT_5X5_100):
        self.k = k_matrix
        self.d = d_coeffs
        self.marker_size = marker_size
        self.model = model
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        # Pre-load marker extrinsics from the model
        self.extrinsics = {face["aruco_id"]: face["extrinsics"] for face in model.extrinsics}

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        return corners, ids

    def estimate_pose(self, corners, ids):
        obj_pts = []
        img_pts = []
        used_ids = []

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.extrinsics:
                    extr = self.extrinsics[marker_id]
                    tvec_model = np.array(extr[:3], dtype=np.float32).reshape(3, 1)
                    rvec_model = np.array(extr[3:], dtype=np.float32).reshape(3, 1)

                    marker_corners = np.array([
                        [-1, 1, 0],
                        [1, 1, 0],
                        [1, -1, 0],
                        [-1, -1, 0]
                    ], dtype=np.float32) * (self.marker_size / 2.0)

                    R_model, _ = cv2.Rodrigues(rvec_model)
                    transformed_corners = (R_model @ marker_corners.T).T + tvec_model.T
                    obj_pts.extend(transformed_corners)
                    img_pts.extend(corners[i][0])
                    used_ids.append(marker_id)

        if len(obj_pts) >= 4:
            obj_pts = np.array(obj_pts, dtype=np.float32)
            img_pts = np.array(img_pts, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.k, self.d, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                return rvec, tvec, used_ids

        return None, None, used_ids