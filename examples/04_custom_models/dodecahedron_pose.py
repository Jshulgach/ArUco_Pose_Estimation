import cv2
import argparse
import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.dodecahedron_model import CleanDodecahedronModel, DodecahedronWithMarkers
from src.utils import legacy_utils as utils
from scipy.optimize import least_squares

from gpt_aruco_pose_estimator import ArucoPoseEstimatorFromModel
from gpt_pose_refiner import PoseRefiner
from gpt_predictive_filter import PredictivePoseFilter

DEBUG = True
CAMERA_ID = 0  # USB camera ID
use_full_pose = False

#aruco_id_order = ['0','1','3','4','5','6','7','8','9','10','11','12']
#aruco_id_to_face_index = {int(mid): idx for idx, mid in enumerate(aruco_id_order)}

class PoseRefiner:
    def __init__(self, camera_matrix, dist_coeffs):
        self.K = camera_matrix
        self.D = dist_coeffs

    def refine_pose(self, obj_pts, img_pts, rvec_init, tvec_init):
        def reprojection_error(params):
            rvec = params[:3].reshape((3, 1))
            tvec = params[3:].reshape((3, 1))
            projected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
            projected_pts = projected_pts.reshape(-1, 2)
            return (projected_pts - img_pts).ravel()

        init_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        result = least_squares(reprojection_error, init_params, method='lm')

        refined_rvec = result.x[:3].reshape((3, 1))
        refined_tvec = result.x[3:].reshape((3, 1))
        return refined_rvec, refined_tvec, result.cost

class DodecahedronPoseTracker:
    def __init__(self, estimator, smoothing=0.8):
        """
        estimator: an instance of ArucoPoseEstimator
        smoothing: float between 0 and 1 (higher = more smoothing)
        """
        self.estimator = estimator
        self.alpha = smoothing
        self.last_rvec = None
        self.last_tvec = None

    def update(self, frame):
        corners, ids = self.estimator.detect(frame)
        rvec, tvec, used_ids = self.estimator.estimate_pose(corners, ids)

        if rvec is not None and tvec is not None:
            # Smooth pose
            if self.last_rvec is not None:
                rvec = self.alpha * self.last_rvec + (1 - self.alpha) * rvec
                tvec = self.alpha * self.last_tvec + (1 - self.alpha) * tvec

            self.last_rvec = rvec
            self.last_tvec = tvec

            return rvec, tvec, used_ids
        else:
            # Return last known pose if available
            return self.last_rvec, self.last_tvec, used_ids

class ArucoDodecahedronPoseEstimator:
    def __init__(self, k_matrix, d_coeffs, marker_size, model, aruco_dict=cv2.aruco.DICT_5X5_100):
        self.k = k_matrix
        self.d = d_coeffs
        self.marker_size = marker_size
        self.model = model
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

        # Pre-index marker poses from model
        self.marker_pose_by_id = {face["aruco_id"]: face["extrinsics"] for face in model.extrinsics}

    def detect_and_estimate_pose(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None, None, [], []

        obj_pts = []
        img_pts = []
        used_ids = []

        print(f"{len(ids)} ids detected")
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.marker_pose_by_id:
                continue

            # Get this marker's pose in model space
            extr = self.marker_pose_by_id[marker_id]
            tvec_model = np.array(extr[:3], dtype=np.float32).reshape(3, 1)
            rvec_model = np.array(extr[3:], dtype=np.float32).reshape(3, 1)
            R_model, _ = cv2.Rodrigues(rvec_model)

            # Define local marker corners in marker frame
            half = self.marker_size / 2
            marker_corners = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0]
            ], dtype=np.float32)

            # Transform marker corners to model space
            transformed_corners = (R_model @ marker_corners.T).T + tvec_model.T
            obj_pts.extend(transformed_corners)
            img_pts.extend(corners[i][0])
            used_ids.append(marker_id)

        print(f"{len(obj_pts)} object points seen (need 4 or more)")
        if len(obj_pts) >= 4:
            obj_pts = np.array(obj_pts, dtype=np.float32)
            img_pts = np.array(img_pts, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.k, self.d, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                return rvec, tvec, corners, ids

        return None, None, corners, ids

def draw_dodecahedron(frame, rvec, tvec, model, K, D, color=(0, 255, 255), thickness=2):
    verts = model.vertices
    faces = model.pentagon_faces

    # Project 3D vertices to 2D
    verts_2d, _ = cv2.projectPoints(verts, rvec, tvec, K, D)
    verts_2d = verts_2d.reshape(-1, 2).astype(int)

    for face in faces:
        pts = verts_2d[face]
        if np.any(pts < 0) or np.any(pts > max(frame.shape[:2])):
            continue
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    return frame

def load_dodecahedron_with_markers(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)["faces"]

def rtvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def collect_pnp_correspondences(model, detected_corners, detected_ids, marker_size):
    """Return matched 3D-2D points for solvePnP."""
    obj_pts = []
    img_pts = []

    marker_data = dict(model.get_marker_corners_with_ids(marker_size))  # id -> 3D corners

    for i, marker_id in enumerate(detected_ids):
        if marker_id in marker_data:
            world_corners = marker_data[marker_id]
            image_corners = detected_corners[i][0]  # shape (4,2)

            obj_pts.extend(world_corners)
            img_pts.extend(image_corners)

    return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)

def draw_dodecahedron_on_image(model, frame, rvec, tvec, camera_matrix, dist_coeffs):
    verts = model.vertices
    faces = model.pentagon_faces

    # Project 3D vertices to 2D
    verts_2d, _ = cv2.projectPoints(verts, rvec, tvec, camera_matrix, dist_coeffs)
    verts_2d = verts_2d.reshape(-1, 2).astype(int)

    # Draw each face
    for face in faces:
        pts = verts_2d[face]
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    return frame

def draw_dodecahedron_on_image2(faces, frame, rvec, tvec, camera_matrix, dist_coeffs):

    for face in faces:
        verts = np.array(face["vertices"])
        aruco_id = face["aruco_id"]
        extrinsics = face["extrinsics"]
        tvec = np.array(extrinsics[:3])
        rvec = np.array(extrinsics[3:])

        # Project 3D vertices to 2D
        verts_2d, _ = cv2.projectPoints(verts, rvec, tvec, camera_matrix, dist_coeffs)
        verts_2d = verts_2d.reshape(-1, 2).astype(int)
        cv2.polylines(frame, [verts_2d], isClosed=True, color=(255, 0, 255), thickness=2)

def draw_full_dodecahedron(frame, rvec, tvec, model, camera_matrix, dist_coeffs):
    for face in model.extrinsics:
        pentagon_3d = np.array(face["vertices"], dtype=np.float32)
        model_rvec = np.array(face["extrinsics"][3:]).reshape(3, 1)
        model_tvec = np.array(face["extrinsics"][:3]).reshape(3, 1)

        R_model, _ = cv2.Rodrigues(model_rvec)
        T_model = np.eye(4)
        T_model[:3, :3] = R_model
        T_model[:3, 3:] = model_tvec
        T_model_inv = np.linalg.inv(T_model)

        # Transform face into marker-local coordinates
        verts_3d_h = np.hstack([pentagon_3d, np.ones((5, 1))]).T
        verts_marker_frame = (T_model_inv @ verts_3d_h).T[:, :3].astype(np.float32)

        # Project into camera frame using estimated dodecahedron pose
        pts_2d, _ = cv2.projectPoints(verts_marker_frame, rvec, tvec, camera_matrix, dist_coeffs)
        pts_2d = pts_2d.astype(int).reshape(-1, 1, 2)

        cv2.polylines(frame, [pts_2d], isClosed=True, color=(255, 0, 255), thickness=2)
        cv2.fillPoly(frame, [pts_2d], color=(255, 0, 255, 64))

def draw_visible_faces(frame, rvec, tvec, model, camera_matrix, dist_coeffs, visible_ids):
    for face in model.extrinsics:
        if face["aruco_id"] not in visible_ids:
            continue

        pentagon_3d = np.array(face["vertices"], dtype=np.float32)
        model_rvec = np.array(face["extrinsics"][3:]).reshape(3, 1)
        model_tvec = np.array(face["extrinsics"][:3]).reshape(3, 1)

        R_model, _ = cv2.Rodrigues(model_rvec)
        T_model = np.eye(4)
        T_model[:3, :3] = R_model
        T_model[:3, 3:] = model_tvec
        T_model_inv = np.linalg.inv(T_model)

        verts_3d_h = np.hstack([pentagon_3d, np.ones((5, 1))]).T
        verts_marker_frame = (T_model_inv @ verts_3d_h).T[:, :3].astype(np.float32)

        pts_2d, _ = cv2.projectPoints(verts_marker_frame, rvec, tvec, camera_matrix, dist_coeffs)
        pts_2d = pts_2d.astype(int).reshape(-1, 1, 2)

        cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.fillPoly(frame, [pts_2d], color=(0, 255, 255, 64))

def draw_faces_per_marker_pose_old(frame, markers, model, k, d):
    for i, marker_id in enumerate(markers['id']):
        rvec, tvec = markers['pose'][i]

        # Try to find the face associated with this marker
        matching_faces = [face for face in model.extrinsics if face["aruco_id"] == marker_id]
        if not matching_faces:
            continue  # No face for this marker

        face = matching_faces[0]
        pentagon_3d = np.array(face["vertices"], dtype=np.float32)

        # Transform face into marker's local coordinate frame
        model_rvec = np.array(face["extrinsics"][3:]).reshape(3, 1)
        model_tvec = np.array(face["extrinsics"][:3]).reshape(3, 1)

        R_model, _ = cv2.Rodrigues(model_rvec)
        T_model = np.eye(4)
        T_model[:3, :3] = R_model
        T_model[:3, 3:] = model_tvec
        T_model_inv = np.linalg.inv(T_model)

        verts_3d_h = np.hstack([pentagon_3d, np.ones((5, 1))]).T
        verts_marker_frame = (T_model_inv @ verts_3d_h).T[:, :3].astype(np.float32)

        # Project the transformed face into the image using that marker's pose
        pts_2d, _ = cv2.projectPoints(verts_marker_frame, rvec, tvec, k, d)
        pts_2d = pts_2d.astype(int).reshape(-1, 1, 2)

        cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.fillPoly(frame, [pts_2d], color=(0, 255, 255, 64))

        # Optional: draw the marker axis
        cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05)

def draw_faces_per_marker_pose(frame, corners, ids, model, k, d, marker_size):
    if ids is None:
        return

    for i, marker_id in enumerate(ids.flatten()):
        matching_faces = [f for f in model.extrinsics if f["aruco_id"] == marker_id]
        if not matching_faces:
            continue

        face = matching_faces[0]
        pentagon_3d = np.array(face["vertices"], dtype=np.float32)

        # Marker pose in model space
        model_rvec = np.array(face["extrinsics"][3:]).reshape(3, 1)
        model_tvec = np.array(face["extrinsics"][:3]).reshape(3, 1)
        R_model, _ = cv2.Rodrigues(model_rvec)

        # Define marker corner layout
        half = marker_size / 2
        marker_corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        # Estimate marker pose using its corners only
        retval, rvec, tvec = cv2.solvePnP(marker_corners, corners[i][0], k, d)
        if not retval:
            continue

        # Transform face into marker-local frame
        T_model = np.eye(4)
        T_model[:3, :3] = R_model
        T_model[:3, 3:] = model_tvec
        T_model_inv = np.linalg.inv(T_model)

        verts_h = np.hstack([pentagon_3d, np.ones((5, 1))]).T
        verts_marker_frame = (T_model_inv @ verts_h).T[:, :3].astype(np.float32)

        pts_2d, _ = cv2.projectPoints(verts_marker_frame, rvec, tvec, k, d)
        pts_2d = pts_2d.astype(int).reshape(-1, 1, 2)

        cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.fillPoly(frame, [pts_2d], color=(0, 255, 255, 64))

        cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05)


def main_new(args, use_full_pose):

    # Load camera intrinsics
    k = np.load(args.k_matrix)
    d = np.load(args.d_coeff)

    # Load the full dodecahedron model with markers
    model = DodecahedronWithMarkers(edge_length=0.025)

    # Setup estimator and tracker
    #estimator = ArucoPoseEstimatorFromModel(k, d, args.aruco_size, model)
    #tracker = DodecahedronPoseTracker(estimator, smoothing=0.8)
    # Pose estimator using marker layout from model
    estimator = ArucoDodecahedronPoseEstimator(
        k_matrix=k,
        d_coeffs=d,
        marker_size=args.aruco_size,
        model=model,
        aruco_dict=cv2.aruco.DICT_5X5_100
    )

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("🔴 Starting Dodecahedron Tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #rvec, tvec, visible_ids = tracker.update(frame)
        rvec, tvec, corners, ids = estimator.detect_and_estimate_pose(frame)

        #if rvec is not None and tvec is not None:
        #    # Draw global axes
        #    cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05)

        #    #draw_dodecahedron(frame, rvec, tvec, model, k, d)
        #    draw_full_dodecahedron(frame, rvec, tvec, model, k, d)
        #    #draw_visible_faces(frame, rvec, tvec, model, k, d, visible_ids)
        print(rvec, tvec)
        if use_full_pose and rvec is not None and tvec is not None:
            cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05)
            #draw_full_dodecahedron(frame, rvec, tvec, model, k, d)
            # Compute and project the 3D center of the dodecahedron model
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)  # model origin
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, k, d)
            center_2d = tuple(center_2d[0].ravel().astype(int))

            # Draw a red dot at the projected center
            cv2.circle(frame, center_2d, radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(frame, "Center", (center_2d[0] + 8, center_2d[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        else:
            draw_faces_per_marker_pose(frame, corners, ids, model, k, d, args.aruco_size)

        cv2.putText(
            frame,
            f"Mode: {'Full Pose' if use_full_pose else 'Per-Marker'} [press 'f' to toggle]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("DodecaPen Tracker", frame)
        key =  cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            use_full_pose = not use_full_pose

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 Tracking complete.")


def main(model, k, d, aruco_type, aruco_size):

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("🔴 Starting Dodecahedron Tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers
        pose_image, markers = utils.pose_estimation_single_camera(frame, aruco_type, aruco_size, k, d)
        #corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if len(markers['id']) > 0:
            # Estimate pose for each marker
            #rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.017, camera_matrix, dist_coeffs)

            center_positions = []

            for i, pose in enumerate(markers["pose"]):
                #if markers["id"][i] == face_data["aruco_id"]:
                marker_id = markers["id"][i]
                rvec = pose[0]
                tvec = pose[1]

                # Draw marker axes
                cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.05, thickness=2)

                # Try to find the face with this ArUco ID
                matching_faces = [face for face in model.extrinsics if face["aruco_id"] == marker_id]
                if not matching_faces:
                    continue  # Skip if marker ID is not part of the dodecahedron
                face_data = matching_faces[0]
                pentagon_3d = np.array(face_data["vertices"], dtype=np.float32)

                # 1. Get local transform of marker on dodecahedron (from model)
                model_rvec = np.array(face_data["extrinsics"][3:]).reshape(3, 1)
                model_tvec = np.array(face_data["extrinsics"][:3]).reshape(3, 1)

                # 2. Convert to transformation matrix
                R_marker_in_model, _ = cv2.Rodrigues(model_rvec)
                T_model_to_marker = np.eye(4)
                T_model_to_marker[:3, :3] = R_marker_in_model
                T_model_to_marker[:3, 3:] = model_tvec

                R_cam_to_marker, _ = cv2.Rodrigues(rvec)
                T_cam_to_marker = np.eye(4)
                T_cam_to_marker[:3, :3] = R_cam_to_marker
                T_cam_to_marker[:3, 3] = tvec

                # 3. Invert to get marker-local transform
                T_marker_to_model = np.linalg.inv(T_model_to_marker)

                # Compute camera→model center transform
                T_cam_to_model = T_cam_to_marker @ T_marker_to_model

                # Store estimated model center in camera coordinates
                center_cam = T_cam_to_model[:3, 3].reshape(3)
                center_positions.append(center_cam)

                # Project the dodecahedron center ([0,0,0] in model space)
                center_in_cam = T_cam_to_model[:3, 3:].T.astype(np.float32)
                center_2d, _ = cv2.projectPoints(center_in_cam, np.zeros((3, 1)), np.zeros((3, 1)), k, d)
                center_2d = tuple(center_2d[0].ravel().astype(int))

                # Draw red dot
                cv2.circle(frame, center_2d, radius=6, color=(0, 0, 255), thickness=-1)
                cv2.putText(frame, "Center", (center_2d[0] + 8, center_2d[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # 4. Convert face vertices to homogeneous and transform to marker-local
                verts_3d_h = np.hstack([pentagon_3d, np.ones((5, 1))]).T  # shape (4, 5)
                verts_marker_frame = (T_marker_to_model @ verts_3d_h).T[:, :3].astype(np.float32)

                # 5. Project into camera image using detected marker pose
                pts_2d, _ = cv2.projectPoints(verts_marker_frame, rvec, tvec, k, d)
                pts_2d = pts_2d.astype(int).reshape(-1, 1, 2)

                # 6. Draw the pentagon
                cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 255, 255), thickness=2)
                #cv2.fillPoly(frame, [pts_2d], color=(0, 255, 255, 64))

            # Only draw if we have estimates from at least 1 marker
            if len(center_positions) > 0:
                avg_center_cam = np.mean(center_positions, axis=0).reshape(1, 3).astype(np.float32)
                center_2d, _ = cv2.projectPoints(avg_center_cam, np.zeros((3, 1)), np.zeros((3, 1)), k, d)
                center_2d = tuple(center_2d[0].ravel().astype(int))

                cv2.circle(frame, center_2d, radius=6, color=(0, 255, 0), thickness=-1)
                cv2.putText(frame, "Avg Center", (center_2d[0] + 8, center_2d[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        #pose_image, markers = utils.pose_estimation_single_camera(frame, aruco_type, aruco_size, k, d)
        #if len(markers['id']) > 0:
            #draw_dodecahedron(frame, rvec_ref, tvec_ref, model, K, D)

            #if len(markers['id']) >= 4:
            #    # Use multi marker solvePnP
            #    obj_pts, img_pts = collect_pnp_correspondences(model, markers['corners'], markers['id'], aruco_size)
            #    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, k, d)
            #    #if success:

            # Get the name of the first marker seen
            #marker_id = markers['id'][0]

            # Get the face data of that marker
            #if marker_id in aruco_id_to_face_index:
            #    face_idx = aruco_id_to_face_index[marker_id]
            #    rvec, tvec = model[face_idx]['extrinsics'][3:], model[face_idx]['extrinsics'][:3]

                # Only draw that polygon
            #    verts = np.array(model[face_idx]["vertices"])
            #    verts_2d, _ = cv2.projectPoints(verts, rvec, tvec, k,d)
            #    verts_2d = verts_2d.reshape(-1, 2).astype(int)
            #    cv2.polylines(pose_image, [verts_2d], isClosed=True, color=(255, 0, 255), thickness=2)



            # The first marker
            #rvec, tvec = markers['pose'][0]

            # The marker has its own rotation and translation from the faces_data than we have to factro in
            #new_rvec = rvec
            #new_tvec = tvec
            #if markers['id'][0] in aruco_id_to_face_index:

            #marker_id = markers['id'][0]
            #pose_image = draw_dodecahedron_on_image2(model, pose_image, rvec, tvec, k, d)
            #pose_image = draw_dodecahedron_on_image(model, pose_image, rvec, tvec, k, d)

            #else:
                # Use single marker as reference
            #    rvec, tvec = markers['pose'][0]
            #    marker_id = markers['id'][0]

                # # Find the corresponding face index
                # if marker_id in aruco_id_to_face_index:
                #     face_idx = aruco_id_to_face_index[marker_id]
                #     # Get pose of that marker in object frame
                #     R_obj, t_obj = model.get_marker_poses()[face_idx]
                #     T_obj_marker = np.eye(4)
                #     T_obj_marker[:3, :3] = R_obj
                #     T_obj_marker[:3, 3] = t_obj.flatten()
                #
                #     # Compute full object pose in camera frame
                #     T_cam_marker = rtvec_to_matrix(rvec, tvec)
                #     T_cam_obj = T_cam_marker @ np.linalg.inv(T_obj_marker)
                #
                #     verts_h = np.hstack([model.vertices, np.ones((len(model.vertices), 1))])  # Nx4
                #     cam_vertices = (T_cam_obj @ verts_h.T).T[:, :3]  # Transform vertices to camera frame
                #     # Project vertices into the image
                #     verts_2d, _ = cv2.projectPoints(cam_vertices, np.zeros((3, 1)), np.zeros((3, 1)), k, d)
                #     verts_2d = verts_2d.reshape(-1, 2).astype(int)
                #
                #     # Project the dodecahedron
                #     for face in model.pentagon_faces:
                #         pts = verts_2d[face]
                #         cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

        cv2.imshow("DodecaPen Tracker", pose_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 Tracking complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--k_matrix", type=str, default='camera_intrinsics.npy', help="Path to calibration matrix (numpy file)")
    parser.add_argument("--d_coeff", type=str, default='camera_distortion.npy', help="Path to distortion coefficients (numpy file)")
    parser.add_argument("--aruco_type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    parser.add_argument("--aruco_size", type=float, default=0.015, help="Size of the ArUco marker in mm")
    args = parser.parse_args()

    #main(args, use_full_pose)
    k = np.load(args.k_matrix)
    d = np.load(args.d_coeff)

    # Generate the model
    #model = CleanDodecahedronModel(edge_length=0.015)
    model = DodecahedronWithMarkers(edge_length=0.025)
    #faces_data = load_dodecahedron_with_markers("dodecahedron_with_markers.json")
    main(model, k, d, args.aruco_type, args.aruco_size)
    #main(face_data, k, d, args.aruco_type, args.aruco_size)
