import numpy as np

def refine_pose(pose_init, model_image, camera_image, J_func, max_iter=10, alpha=0.5):
    p = pose_init.copy()
    for _ in range(max_iter):
        residuals, J = J_func(p, model_image, camera_image)
        delta_p = np.linalg.pinv(J.T @ J) @ J.T @ residuals
        p_new = p + alpha * delta_p
        if np.linalg.norm(delta_p) < 1e-4:
            break
        p = p_new
    return p


def draw_dodecahedron_overlay(image, model, K, R=None, t=None, draw_ids=True):
    if R is None: R = np.eye(3)
    if t is None:
        t = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # pulls the model in front of camera

    verts = model.vertices
    centers = model.get_face_centers()
    faces = model.pentagon_faces
    print(verts)
    print(centers)
    print(faces)

    # Project 3D points to 2D
    rvec, _ = cv2.Rodrigues(R)
    projected_verts, _ = cv2.projectPoints(verts, rvec, t, K, np.zeros((5, 1)))
    projected_verts = projected_verts.squeeze().astype(int)

    # Project centers too
    projected_centers, _ = cv2.projectPoints(centers, rvec, t, K, np.zeros((5, 1)))
    projected_centers = projected_centers.squeeze().astype(int)

    # Draw edges
    for idx, face in enumerate(faces):
        pts = projected_verts[face]
        cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        # Optionally draw face centers and ID labels
        if draw_ids:
            c = tuple(projected_centers[idx])
            cv2.circle(image, c, 3, (0, 0, 255), -1)
            cv2.putText(image, f"ID {idx}", (c[0] + 5, c[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return image

def estimate_pose_kabsch(object_points, camera_points):
    object_points = np.array(object_points)
    camera_points = np.array(camera_points)

    centroid_obj = np.mean(object_points, axis=0)
    centroid_cam = np.mean(camera_points, axis=0)

    X = object_points - centroid_obj
    Y = camera_points - centroid_cam

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_cam - R @ centroid_obj
    return R, t.reshape(3, 1)
class PoseKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 6)  # [rvec, tvec] in state + measurement
        self.kf.measurementMatrix = np.eye(6, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def update(self, rvec, tvec):
        measurement = np.vstack((rvec.astype(np.float32), tvec.astype(np.float32)))
        self.kf.correct(measurement)
        pred = self.kf.predict()
        rvec_filtered = pred[:3]
        tvec_filtered = pred[3:]
        return rvec_filtered, tvec_filtered

def estimate_pose_from_individual_markers(model, corners, ids, camera_matrix, dist_coeffs, marker_size):
    if ids is None or len(ids) == 0:
        return False, None, None, None

    retval, rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
    object_points = []
    camera_points = []

    poses = model.get_marker_poses()

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in aruco_id_to_face_index:
            continue

        face_idx = aruco_id_to_face_index[marker_id]
        T_obj_marker = np.eye(4)
        R_obj, t_obj = poses[face_idx]
        T_obj_marker[:3, :3] = R_obj
        T_obj_marker[:3, 3] = t_obj.flatten()
        marker_center_obj = T_obj_marker[:3, 3]

        T_cam_marker = rtvec_to_matrix(rvecs[i], tvecs[i])
        marker_center_cam = T_cam_marker[:3, 3]

        object_points.append(marker_center_obj)
        camera_points.append(marker_center_cam)

    if len(object_points) >= 3:
        R_est, t_est = estimate_pose_kabsch(object_points, camera_points)
        center_obj = np.zeros((3, 1))
        center_cam = R_est @ center_obj + t_est
        return True, R_est, t_est, center_cam.flatten()

    return False, None, None, None
def estimate_pose_from_dodecahedron(model, corners, ids, camera_matrix, dist_coeffs, marker_size=10.8):
    object_points = []
    image_points = []

    marker_corners = model.get_marker_corners(marker_size=marker_size)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            #if marker_id >= len(marker_corners):
            #    continue  # skip invalid marker ID
            if marker_id not in aruco_id_to_face_index:
                continue

            face_idx = aruco_id_to_face_index[marker_id]
            obj_pts = marker_corners[face_idx]

            #obj_pts = marker_corners[marker_id]  # 4x3 array
            img_pts = corners[i][0]              # 4x2 array

            object_points.extend(obj_pts)
            image_points.extend(img_pts)

        if len(object_points) >= 4:
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                center_3d = np.zeros((1, 3), dtype=np.float32)  # origin of the object
                center_in_camera, _ = cv2.projectPoints(center_3d, rvec, tvec, camera_matrix, dist_coeffs)
                return success, rvec, tvec, center_in_camera[0][0]

    return False, None, None, None

def overlay_virtual_dodecahedron(frame, model, corners, ids, k_matrix, d_coeffs, marker_size):
    """
    Superimpose a virtual dodecahedron model on top of the real one using the pose of a reference marker.
    """
    if ids is None or len(ids) == 0:
        return frame

    # Estimate poses of all visible markers
    retval, rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, k_matrix, d_coeffs)

    # Choose the first visible marker as reference
    ref_idx = 0
    marker_id = ids[ref_idx][0]
    if marker_id not in aruco_id_to_face_index:
        return frame

    face_idx = aruco_id_to_face_index[marker_id]

    # Get pose of reference marker in camera frame
    ref_rvec = rvecs[ref_idx]
    ref_tvec = tvecs[ref_idx]
    T_cam_marker = rtvec_to_matrix(ref_rvec, ref_tvec)

    # Get pose of that marker in object frame
    R_obj, t_obj = model.get_marker_poses()[face_idx]
    T_obj_marker = np.eye(4)
    T_obj_marker[:3, :3] = R_obj
    T_obj_marker[:3, 3] = t_obj.flatten()

    # Compute full object pose in camera frame
    T_marker_obj = np.linalg.inv(T_obj_marker)
    T_cam_obj = T_cam_marker @ T_marker_obj

    # Transform model vertices
    vertices = model.vertices
    vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])  # Nx4
    cam_vertices = (T_cam_obj @ vertices_h.T).T[:, :3]

    # Project vertices into the image
    projected, _ = cv2.projectPoints(cam_vertices, np.zeros((3, 1)), np.zeros((3, 1)), k_matrix, d_coeffs)
    projected = projected.reshape(-1, 2).astype(int)

    # Draw pentagon faces
    for face in model.pentagon_faces:
        pts = projected[face]
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    return frame
