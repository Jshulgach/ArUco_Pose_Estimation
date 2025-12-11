import cv2
import numpy as np
from aruco_pose_pipeline import ArucoPoseEstimator
from dodecahedron_model import CleanDodecahedronModel

# Load your camera calibration
K = np.load("camera_intrinsics.npy")
D = np.load("camera_distortion.npy")

# Initialize pose estimator with ArUco group JSON
estimator = ArucoPoseEstimator(K, D, tag_data_path="aruco_group.json")
model = CleanDodecahedronModel(edge_length=0.025)

cap = cv2.VideoCapture(0)  # or path to video file

def draw_dodecahedron_overlay(frame, model, rvec, tvec, camera_matrix, dist_coeffs):
    verts = model.vertices
    faces = model.pentagon_faces
    verts_2d, _ = cv2.projectPoints(verts, rvec, tvec, camera_matrix, dist_coeffs)
    verts_2d = verts_2d.reshape(-1, 2).astype(int)

    for face in faces:
        pts = verts_2d[face]
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    return frame

def draw_marker_face_overlay(frame, model, rvec, tvec, camera_matrix, dist_coeffs, face_indices, offset_mm=7.0):
    verts = model.vertices
    #face = model.pentagon_faces[face_index]
    for face_index in face_indices:
        face = model.pentagon_faces[face_index]
        pts3d = verts[face]

        # Offset direction: normal to the face
        v1 = pts3d[1] - pts3d[0]
        v2 = pts3d[2] - pts3d[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        offset = -offset_mm / 1000.0 * normal  # convert mm to meters and offset backwards
        pts3d_offset = pts3d + offset

        pts2d, _ = cv2.projectPoints(pts3d_offset, rvec, tvec, camera_matrix, dist_coeffs)
        pts2d = pts2d.reshape(-1, 2).astype(int)
        cv2.polylines(frame, [pts2d], isClosed=True, color=(0, 255, 0), thickness=2)
    return frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rvec, tvec, corners, ids = estimator.detect_and_estimate_pose(frame)

    # Draw all detected marker corners
    if ids is not None:
        for i in [0, 1]:
        # Just draw the ID0 marker
        #if 0 in ids:
            #i = np.where(ids == 0)[0][0]
            # Draw the detected marker corners for ID 0
            cv2.aruco.drawDetectedMarkers(frame, corners[i:i+1], ids[i:i+1])
        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if rvec is not None:
        #frame = estimator.draw_debug_overlay(frame, rvec, tvec)
        frame = draw_marker_face_overlay(frame, model, rvec, tvec, K, D, face_indices=[0, 1], offset_mm=0.0)
        #frame = draw_dodecahedron_overlay(frame, model, rvec, tvec, K, D)

    cv2.imshow("ArUco Dodecahedron Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
