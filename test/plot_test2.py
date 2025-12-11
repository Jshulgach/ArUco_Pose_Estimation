#plot_test2.py
from dodecahedron_model import CleanDodecahedronModel
import matplotlib.pyplot as plt
import numpy as np

model = CleanDodecahedronModel(edge_length=15)
verts = model.vertices
faces = model.pentagon_faces
centers = model.get_face_centers()
marker_corners = model.get_marker_corners(marker_size=10)

#print("Total number of faces (pentagons):", len(faces), "data:" , faces)  # Debugging info
#print(f"Number of centers: {len(centers)}, data: {centers}")
#print(f"Vertices shape: {verts.shape}, data: {verts[:5]}")  # Show first 5 vertices for debugging
#print("Face centers:", centers.shape)
#print("First center:", centers[0])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='k', s=5)

# Add the number label to each vertex for debugging purposes
for i, (x, y, z) in enumerate(verts):
    ax.text(x, y, z, str(i), color='red', fontsize=8, zorder=5)  # Label vertices with their index


# Set proper aspect ratio and limits
max_range = (verts.max(axis=0) - verts.min(axis=0)).max()
mid = verts.mean(axis=0)
for axis, m in zip('xyz', mid):
    getattr(ax, f'set_{axis}lim')(m - max_range / 2, m + max_range / 2)

for face, center in zip(faces, centers):
    face_verts = [verts[i] for i in face]
    ax.plot([center[0]], [center[1]], [center[2]], marker='o', markersize=8, color='red')

# Plot marker corners
for idx, corners in enumerate(marker_corners):
    # Draw red square for each marker
    square = np.vstack([corners, corners[0]])  # loop back to start
    ax.plot(square[:, 0], square[:, 1], square[:, 2], color='green', linewidth=1)

    # Label marker ID at center
    center = centers[idx]
    ax.text(center[0], center[1], center[2], str(idx), color='blue', fontsize=8)

# Force origin marker
ax.plot([0], [0], [0], marker='o', color='blue', markersize=8, label='origin')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Dodecahedron Model with Marker Corners and Face Centers")
plt.show()
