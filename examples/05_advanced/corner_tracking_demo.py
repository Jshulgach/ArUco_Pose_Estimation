import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.dodecahedron_model import DodecahedronWithMarkers
from src.utils import legacy_utils as utils
from scipy.optimize import least_squares

CAMERA_ID = 0  # USB camera ID

class CornerTracker:
    def __init__(self, aruco_dict, aruco_params, interval=5):
        self.aruco_dict = aruco_dict
        self.aruco_params = aruco_params
        self.interval = interval
        self.counter = 0
        self.prev_gray = None
        self.prev_corners = None
        self.prev_ids = None

    def detect_or_track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            self.prev_corners = corners
            self.prev_ids = ids
            self.prev_gray = gray.copy()
            return corners, ids

        # If no detection, try tracking previous corners
        if self.prev_corners is not None and self.prev_gray is not None:
            tracked_corners = []
            valid_ids = []
            for c, mid in zip(self.prev_corners, self.prev_ids):
                nextPts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, c, None)
                if status.sum() == 4:
                    tracked_corners.append(nextPts)
                    valid_ids.append(mid)
            if len(tracked_corners) > 0:
                self.prev_corners = tracked_corners
                self.prev_ids = np.array(valid_ids)
                self.prev_gray = gray.copy()
                return tracked_corners, self.prev_ids

        return [], []

def refine_pose_gauss_newton(k, d, obj_pts, img_pts, rvec_init, tvec_init):
    def reprojection_error(params):
        rvec = params[:3].reshape((3, 1))
        tvec = params[3:].reshape((3, 1))
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, d)
        return (proj_pts.reshape(-1, 2) - img_pts.reshape(-1, 2)).ravel()

    init_params = np.hstack((rvec_init.ravel(), tvec_init.ravel()))
    result = least_squares(reprojection_error, init_params, method='trf', max_nfev=20)
    refined_rvec = result.x[:3].reshape(3, 1)
    refined_tvec = result.x[3:].reshape(3, 1)
    return refined_rvec, refined_tvec

def extract_marker_patch(frame, corners, size=100):
    # corners must be np.array with shape (4, 2), in order: TL, TR, BR, BL
    dst_corners = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners, dst_corners)
    marker_patch = cv2.warpPerspective(frame, H, (size, size))
    return marker_patch

def detect_marker_from_patch(patch, aruco_dict, parameters):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray_patch, aruco_dict, parameters=parameters)
    return ids, corners

def get_marker_corners_from_face(face, marker_size):
    half = marker_size / 2.0
    marker_corners_local = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    tvec_model = np.array(face["extrinsics"][:3], dtype=np.float32).reshape(3, 1)
    rvec_model = np.array(face["extrinsics"][3:], dtype=np.float32).reshape(3, 1)
    R_model, _ = cv2.Rodrigues(rvec_model)

    corners_3d = (R_model @ marker_corners_local.T).T + tvec_model.T
    return corners_3d


def main(model, k, d, aruco_type, aruco_size):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_type))
    aruco_params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, 'DetectorParameters_create') else cv2.aruco.DetectorParameters()
    tracker = CornerTracker(aruco_dict, aruco_params)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 0
    REFINE_INTERVAL = 5
    latest_rvec = None
    latest_tvec = None

    print("🔴 Starting Dodecahedron Tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids = tracker.detect_or_track(frame)
        
        # Only try patch-based recovery if ArUco detection failed
        if corners is not None and (ids is None or len(ids) == 0):
            for c in corners:
                quad = np.array(c[0], dtype=np.float32)
                if quad.shape[0] == 4:
                    patch = extract_marker_patch(frame, quad)
                    detected_ids, _ = detect_marker_from_patch(patch, aruco_dict, aruco_params)
                    if detected_ids is not None:
                        cv2.putText(frame, f"Recovered ID: {detected_ids[0][0]}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if corners is not None:
            for marker in corners:
                for corner in marker[0]:
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

        #pose_image, markers = utils.pose_estimation_single_camera(frame, aruco_type, aruco_size, k, d)

        # obj_pts_all = []
        # img_pts_all = []
        # if len(markers['id']) > 0:
        #     center_positions = []
        #     for i, pose in enumerate(markers["pose"]):
        #         marker_id = markers["id"][i]
        #         rvec = pose[0]
        #         tvec = pose[1]
        #
        #         #cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05, thickness=2)
        #
        #         matching_faces = [face for face in model.extrinsics if face["aruco_id"] == marker_id]
        #         if not matching_faces:
        #             continue
        #         face_data = matching_faces[0]
        #
        #         obj_pts = np.array(get_marker_corners_from_face(face_data, aruco_size), dtype=np.float32)
        #         if ids is not None:
        #             match_idx = np.where(ids.flatten() == marker_id)[0]
        #             if len(match_idx) == 0:
        #                 continue  # No matching corner found
        #             corner_idx = match_idx[0]
        #             img_pts = np.array(corners[corner_idx][0], dtype=np.float32)
        #         else:
        #             continue
        #
        #         obj_pts_all.append(obj_pts)
        #         img_pts_all.append(img_pts)
        #
        #         pentagon_3d = np.array(face_data["vertices"], dtype=np.float32)
        #         model_rvec = np.array(face_data["extrinsics"][3:]).reshape(3, 1)
        #         model_tvec = np.array(face_data["extrinsics"][:3]).reshape(3, 1)
        #         R_marker_in_model, _ = cv2.Rodrigues(model_rvec)
        #         T_model_to_marker = np.eye(4)
        #         T_model_to_marker[:3, :3] = R_marker_in_model
        #         T_model_to_marker[:3, 3:] = model_tvec
        #
        #         R_cam_to_marker, _ = cv2.Rodrigues(rvec)
        #         T_cam_to_marker = np.eye(4)
        #         T_cam_to_marker[:3, :3] = R_cam_to_marker
        #         T_cam_to_marker[:3, 3] = tvec
        #
        #         T_marker_to_model = np.linalg.inv(T_model_to_marker)
        #         T_cam_to_model = T_cam_to_marker @ T_marker_to_model
        #         center_cam = T_cam_to_model[:3, 3].reshape(3)
        #         center_positions.append(center_cam)
        #
        #         center_in_cam = T_cam_to_model[:3, 3:].T.astype(np.float32)
        #         center_2d, _ = cv2.projectPoints(center_in_cam, np.zeros((3, 1)), np.zeros((3, 1)), k, d)
        #         center_2d = tuple(center_2d[0].ravel().astype(int))
        #         cv2.circle(frame, center_2d, radius=6, color=(0, 0, 255), thickness=-1)
        #         cv2.putText(frame, "Center", (center_2d[0] + 8, center_2d[1] - 8),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #
        #         verts_3d_h = np.hstack([pentagon_3d, np.ones((5, 1))]).T
        #         verts_marker_frame = (T_marker_to_model @ verts_3d_h).T[:, :3].astype(np.float32)
        #         pts_2d, _ = cv2.projectPoints(verts_marker_frame, rvec, tvec, k, d)
        #         pts_2d = pts_2d.astype(int).reshape(-1, 1, 2)
        #         cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 255, 255), thickness=2)
        #
        #     if len(center_positions) > 0:
        #         avg_center_cam = np.mean(center_positions, axis=0).reshape(1, 3).astype(np.float32)
        #         center_2d, _ = cv2.projectPoints(avg_center_cam, np.zeros((3, 1)), np.zeros((3, 1)), k, d)
        #         center_2d = tuple(center_2d[0].ravel().astype(int))
        #         cv2.circle(frame, center_2d, radius=6, color=(0, 255, 0), thickness=-1)
        #         cv2.putText(frame, "Avg Center", (center_2d[0] + 8, center_2d[1] - 8),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #
        #
        # # Flatten and optimize once
        # if len(obj_pts_all) > 0 and frame_count % REFINE_INTERVAL == 0:
        #     obj_pts_all = np.vstack(obj_pts_all)
        #     img_pts_all = np.vstack(img_pts_all)
        #     success, rvec, tvec = cv2.solvePnP(obj_pts_all, img_pts_all, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
        #     if success:
        #         latest_rvec = rvec
        #         latest_tvec = tvec
        #
        #         #rvec, tvec = refine_pose_gauss_newton(k, d, obj_pts_all, img_pts_all, rvec, tvec)
        #         cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05, 2)
        #
        #         center_3d = np.array([[0, 0, 0]], dtype=np.float32)
        #         center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, k, d)
        #         center_2d = tuple(center_2d[0].ravel().astype(int))
        #         cv2.circle(frame, center_2d, 6, (0, 255, 0), -1)
        #         cv2.putText(frame, "Optimized Center", (center_2d[0] + 8, center_2d[1] - 8),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #
        # # Always draw latest known pose if available
        #     if latest_rvec is not None and latest_tvec is not None:
        #         cv2.drawFrameAxes(frame, k, d, latest_rvec, latest_tvec, 0.05, 2)
        #
        # frame_count += 1

        cv2.imshow("DodecaPen Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 Tracking complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_matrix", type=str, default='camera_intrinsics.npy')
    parser.add_argument("--d_coeff", type=str, default='camera_distortion.npy')
    parser.add_argument("--aruco_type", type=str, default="DICT_5X5_100")
    parser.add_argument("--aruco_size", type=float, default=0.015)
    args = parser.parse_args()

    k = np.load(args.k_matrix)
    d = np.load(args.d_coeff)
    model = DodecahedronWithMarkers(edge_length=0.025)
    main(model, k, d, args.aruco_type, args.aruco_size)
