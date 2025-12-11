import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_dodecahedron_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    face_vertices = []
    aruco_ids = []
    all_vertices = []

    for entry in data:
        vertices = np.array(entry["vertices"])
        face_vertices.append(vertices)
        aruco_ids.append(entry["aruco_id"])
        all_vertices.extend(vertices)

    all_vertices = np.unique(np.array(all_vertices), axis=0)
    return {
        "all_vertices": all_vertices,
        "face_vertices": face_vertices,
        "aruco_ids": aruco_ids
    }

def plot_dodecahedron_model(model_data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face
    for face, aruco_id in zip(model_data["face_vertices"], model_data["aruco_ids"]):
        poly = Poly3DCollection([face], alpha=0.5, facecolor='lightblue', edgecolor='k')  # Create a Poly3DCollection for the face
        poly.set_facecolor('skyblue')
        ax.add_collection3d(poly)

        # Plot all face vertices
        ax.scatter(face[:, 0], face[:, 1], face[:, 2], color='k', s=10)

        # Label face with its ArUco ID at its centroid
        center = np.mean(face, axis=0)
        ax.text(*center, f'{aruco_id}', color='black', fontsize=10, ha='center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Dodecahedron Model with ArUco Marker IDs")
    ax.auto_scale_xyz([-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03])
    plt.tight_layout()
    plt.show()


def plot_dodecahedron_with_marker_axes(model_data, axis_len=0.01):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for face, aruco_id in zip(model_data["face_vertices"], model_data["aruco_ids"]):
        # Draw face polygon
        poly = Poly3DCollection([face], alpha=0.5, facecolor='skyblue', edgecolor='k')
        ax.add_collection3d(poly)
        ax.scatter(face[:, 0], face[:, 1], face[:, 2], color='k', s=10)

        # Face center and normal (Z axis)
        center = np.mean(face, axis=0)
        v1 = face[1] - face[0]
        v2 = face[2] - face[0]
        z_axis = np.cross(v1, v2)
        z_axis /= np.linalg.norm(z_axis)

        # X axis: aligned with first edge
        x_axis = v1 / np.linalg.norm(v1)
        y_axis = np.cross(z_axis, x_axis)

        # Plot marker local axes at face center
        ax.quiver(*center, *(axis_len * x_axis), color='r', label='X' if aruco_id == 0 else "")
        ax.quiver(*center, *(axis_len * y_axis), color='g', label='Y' if aruco_id == 0 else "")
        ax.quiver(*center, *(axis_len * z_axis), color='b', label='Z' if aruco_id == 0 else "")

        ax.text(*center, f'{aruco_id}', color='black', fontsize=9, ha='center')

    ax.set_title("Dodecahedron with ArUco Marker Axes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03])
    plt.legend()
    plt.tight_layout()
    plt.show()


# Load the model from the JSON file
loaded_model = load_dodecahedron_from_json("dodecahedron_face_map.json")

# Visualize the loaded model
#plot_dodecahedron_model(loaded_model)
plot_dodecahedron_with_marker_axes(loaded_model)
