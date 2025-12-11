# visualize_dodecahedron.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dodecahedron_model import DodecahedronWithMarkers
import numpy as np
import cv2

def plot_dodecahedron(dodeca, axis_len=0.01):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for face in dodeca.extrinsics:
        verts = face["vertices"]
        tvec = face["extrinsics"][:3]
        rvec = face["extrinsics"][3:]
        aruco_id = face["aruco_id"]

        poly = Poly3DCollection([verts], alpha=0.4, facecolor='lightblue', edgecolor='k')
        ax.add_collection3d(poly)

        R, _ = cv2.Rodrigues(rvec)
        origin = tvec
        ax.quiver(*origin, *R[:, 0]*axis_len, color='r')
        ax.quiver(*origin, *R[:, 1]*axis_len, color='g')
        ax.quiver(*origin, *R[:, 2]*axis_len, color='b')
        ax.text(*origin, f"ID {aruco_id}", fontsize=8, ha='center')

    ax.set_title("Dodecahedron with ArUco Marker Extrinsics")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.04, 0.04], [-0.04, 0.04], [-0.04, 0.04])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = DodecahedronWithMarkers()
    plot_dodecahedron(model)
