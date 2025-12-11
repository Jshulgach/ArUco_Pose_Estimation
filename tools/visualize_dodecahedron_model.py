# This script is used to visualize the dodecahedron with ArUco markers.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dodecahedron_model import CleanDodecahedronModel

# Create the model for the dodecahedron
model = CleanDodecahedronModel()
verts = model.vertices
faces = model.pentagon_faces
centers = model.get_face_centers()
aruco_ids = model.aruco_id_order

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='k', s=5)

for idx, face in enumerate(faces):
    # Draw dodecahedron faces
    face_verts = [verts[i] for i in face]
    ax.add_collection3d(Poly3DCollection([face_verts], alpha=0.3, facecolor='lightblue', edgecolor='k'))

    # Label each face center with its ArUco marker ID
    center = centers[idx]
    ax.text(center[0], center[1], center[2], f"ID {aruco_ids[idx]}", color='red', fontsize=10)

    # Plot a blue dot at each face center
    ax.scatter(center[0], center[1], center[2], color='b', s=5, label=f'Center {idx}' if idx == 0 else "")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Verify ArUco Tag Layout on Dodecahedron")
plt.show()