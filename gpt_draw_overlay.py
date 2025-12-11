import cv2
import numpy as np

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