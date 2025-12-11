import cv2
import json
import numpy as np
from dodecahedron_model import CleanDodecahedronModel

def rotation_matrix_z(degrees):
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# Optional: rotation correction per face (degrees around face Z)
rotation_offsets = {0: 18,
                    1: -18,
                    3: -18,
                    4: -18,
                    5: -80,
                    6: -90,
                    7: -126,
                    8: -62,
                    9: -62,
                    10: -62,
                    11: -126,
                    12: 0}  # You can specify rotation offsets for specific faces here

model = CleanDodecahedronModel(edge_length=0.017)
faces_output = []

for idx, face in enumerate(model.pentagon_faces):
    aruco_id = int(model.aruco_id_order[idx])
    face_verts = model.vertices[face]
    center = np.mean(face_verts, axis=0)

    # Coordinate frame for marker
    x_axis = face_verts[1] - face_verts[0]
    x_axis /= np.linalg.norm(x_axis)
    v2 = face_verts[2] - face_verts[0]
    z_axis = np.cross(x_axis, v2)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    # Optional Z-axis correction
    correction = rotation_matrix_z(rotation_offsets.get(aruco_id, 0))
    R = R @ correction

    rvec, _ = cv2.Rodrigues(R)
    extrinsics = list(center) + list(rvec.flatten())

    faces_output.append({
        "face_index": idx,
        "aruco_id": aruco_id,
        "vertices": face_verts.tolist(),
        "center": center.tolist(),
        "extrinsics": extrinsics
    })

combined_data = {"faces": faces_output}
output_path = "dodecahedron_with_markers.json"
with open(output_path, "w") as f:
    json.dump(combined_data, f, indent=2)

    print(f"✅ ArUco group JSON saved to {output_path}")
