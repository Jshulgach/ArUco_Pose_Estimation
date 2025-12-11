# visualize_dodecahedron.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dodecahedron_model import DodecahedronModel
import numpy as np

def sort_face_vertices(vertices):
    """Sort 3D face vertices in counter-clockwise order projected onto best-fit plane."""
    center = vertices.mean(axis=0)
    vecs = vertices - center
    normal = np.cross(vecs[1] - vecs[0], vecs[2] - vecs[0])
    normal /= np.linalg.norm(normal)
    # Create a local 2D plane
    x_axis = vecs[0] / np.linalg.norm(vecs[0])
    y_axis = np.cross(normal, x_axis)
    coords_2d = np.dot(vecs, np.stack([x_axis, y_axis], axis=1))
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    return vertices[np.argsort(angles)]

def plot_dodecahedron(model: DodecahedronModel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    verts = model.vertices
    faces = model.faces

    # Plot faces
    for face in faces:
        face_vertices = verts[face]
        ordered_vertices = sort_face_vertices(face_vertices)
        poly = Poly3DCollection([ordered_vertices], alpha=0.4)
        poly.set_edgecolor('k')
        poly.set_facecolor('lightblue')
        ax.add_collection3d(poly)

        # Mark the center of each face (optional)
        center = face_vertices.mean(axis=0)
        ax.scatter(*center, color='red')
        ax.text(*center, f'{faces.index(face)}', fontsize=8, color='black')

    # Plot the vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='black')

    # Equal aspect ratio
    max_range = np.ptp(verts, axis=0).max()
    mid = verts.mean(axis=0)
    for axis, m in zip('xyz', mid):
        getattr(ax, f'set_{axis}lim')(m - max_range / 2, m + max_range / 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Dodecahedron Model with Marker Faces")
    plt.show()


if __name__ == "__main__":
    model = DodecahedronModel()
    plot_dodecahedron(model)
