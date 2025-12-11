import json
import cv2
import time
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packaging.version import Version


def load_aruco_group(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['tags']

def load_dodecahedron_with_markers(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)["faces"]

def plot_marker_frames(marker_data, axis_len=0.01):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for tag_id, tag in marker_data.items():
        tvec = np.array(tag['extrinsics'][:3], dtype=np.float32)
        rvec = np.array(tag['extrinsics'][3:], dtype=np.float32)

        R, _ = cv2.Rodrigues(rvec)
        origin = tvec
        x_axis = R[:, 0] * axis_len
        y_axis = R[:, 1] * axis_len
        z_axis = R[:, 2] * axis_len

        ax.quiver(*origin, *x_axis, color='r', label='X' if tag_id == "0" else "")
        ax.quiver(*origin, *y_axis, color='g', label='Y' if tag_id == "0" else "")
        ax.quiver(*origin, *z_axis, color='b', label='Z' if tag_id == "0" else "")
        ax.text(*origin, f"ID {tag_id}", fontsize=9, ha='center')

    ax.set_title("ArUco Marker Extrinsics Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.04, 0.04], [-0.04, 0.04], [-0.04, 0.04])
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_aruco_marker_image(marker_id, marker_size=100, dictionary=cv2.aruco.DICT_5X5_100):

    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)

    # Support for multiple cv2 versions
    if Version(cv2.__version__) >= Version("4.7.0"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(dictionary)
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_img, 1)

    return marker_img

def plot_dodecahedron_with_extrinsics(faces, axis_len=0.01):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        verts = np.array(face["vertices"])
        aruco_id = face["aruco_id"]
        extrinsics = face["extrinsics"]
        tvec = np.array(extrinsics[:3])
        rvec = np.array(extrinsics[3:])

        # Draw the face
        poly = Poly3DCollection([verts], alpha=0.4, facecolor='lightblue', edgecolor='k')
        ax.add_collection3d(poly)

        # Draw marker coordinate frame
        R, _ = cv2.Rodrigues(rvec)
        origin = tvec
        x_axis = R[:, 0] * axis_len
        y_axis = R[:, 1] * axis_len
        z_axis = R[:, 2] * axis_len

        ax.quiver(*origin, *x_axis, color='r')
        ax.quiver(*origin, *y_axis, color='g')
        ax.quiver(*origin, *z_axis, color='b')
        ax.text(*origin, f"ID {aruco_id}", fontsize=9, ha='center')

        # Draw a square on the face with the origin as the square center
        square_size = 0.005

        # Make sure the rotation is applied to the square vertices
        square_verts = np.array([
            [-square_size, -square_size, 0],
            [ square_size, -square_size, 0],
            [ square_size,  square_size, 0],
            [-square_size,  square_size, 0]
        ])
        # Apply rotation
        square_verts_rotated = (R @ square_verts.T).T + origin
        # Draw the square
        poly_square = Poly3DCollection([square_verts_rotated], alpha=0.5, facecolor='yellow', edgecolor='k')
        ax.add_collection3d(poly_square)

    ax.set_title("Dodecahedron with ArUco Marker Frames")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.04, 0.04], [-0.04, 0.04], [-0.04, 0.04])
    plt.tight_layout()
    plt.show()

def generate_checkerboard(n=4, color1=1.0, color2=0.0):
    board = np.indices((n, n)).sum(axis=0) % 2
    board = board.astype(float)
    board[board == 1] = color1
    board[board == 0] = color2
    return board


def plot_dodecahedron_with_textured_markers(faces, axis_len=0.01):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        verts = np.array(face["vertices"])
        aruco_id = face["aruco_id"]
        extrinsics = face["extrinsics"]
        tvec = np.array(extrinsics[:3])
        rvec = np.array(extrinsics[3:])

        # Get ArUco marker image
        marker_img = generate_aruco_marker_image(aruco_id, marker_size=100)
        norm = Normalize(vmin=0, vmax=255)
        rgba = cm.gray(norm(marker_img))

        # Use only 4 vertices of the face to form a quad
        quad = verts[:4]
        x = quad[:, 0].reshape((2, 2))
        y = quad[:, 1].reshape((2, 2))
        z = quad[:, 2].reshape((2, 2))

        # Repeat RGBA pattern to match 2x2 grid required by plot_surface
        texture = np.zeros((2, 2, 4))
        texture[:, :] = rgba[50, 50]  # approximate color from center of marker

        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=texture, shade=False)



        #cv2.imshow("Marker Image", marker_img)  # For debugging, can be removed in production
        #key = cv2.waitKey(1) & 0xFF
        #if key == ord('q'):
        #    pass


        # Assume verts[0:4] approximate a flat quad (for rendering purposes)
        #quad = verts[:4]
        #x = quad[:, 0].reshape((2, 2))
        #y = quad[:, 1].reshape((2, 2))
        #z = quad[:, 2].reshape((2, 2))

        #ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=marker_img, shade=False)

        # Also draw the marker's coordinate frame
        R, _ = cv2.Rodrigues(rvec)
        x_axis = R[:, 0] * axis_len
        y_axis = R[:, 1] * axis_len
        z_axis = R[:, 2] * axis_len

        ax.quiver(*tvec, *x_axis, color='r')
        ax.quiver(*tvec, *y_axis, color='g')
        ax.quiver(*tvec, *z_axis, color='b')
        ax.text(*tvec, f"ID {aruco_id}", fontsize=9, ha='center')

    ax.set_title("Dodecahedron with Real ArUco Markers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.04, 0.04], [-0.04, 0.04], [-0.04, 0.04])
    plt.tight_layout()
    plt.show()

def plot_dodecahedron_with_checkerboards(faces, grid_size=4, square_size=0.01):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        verts = np.array(face["vertices"])
        aruco_id = face["aruco_id"]
        extrinsics = face["extrinsics"]
        tvec = np.array(extrinsics[:3])
        rvec = np.array(extrinsics[3:])

        # Draw the face
        poly = Poly3DCollection([verts], alpha=0.4, facecolor='lightblue', edgecolor='k')
        ax.add_collection3d(poly)

        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='k', s=10)

        ax.text(*tvec, f"ID {aruco_id}", color='red', fontsize=10, ha='center')

        # Compute face coordinate frame
        R, _ = cv2.Rodrigues(rvec)
        x_axis = R[:, 0]
        y_axis = R[:, 1]

        # Generate the aruco image on top of the checkerboard, and convert the image to a 5x5 marker for visualization purposes
        marker_img = generate_aruco_marker_image(aruco_id, marker_size=100)

        # Show the marker image for debugging purposes (optional)
        cv2.imshow("Marker Image", marker_img)
        cv2.waitKey(1)  # Allow the image to be displayed for a short time
        #time.sleep(1)

         # Convert the 100x100 image into the 5x5 marker format for visualization purposes
        marker_img_small = cv2.resize(marker_img, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)

        # Reshow the marker to make sure it is correct after resizing
        print(marker_img_small)
        #time.sleep(10)  # Allow the image to be displayed for a short time

        # Pad marker to match the black background
        #marker_img_small = np.pad(marker_img_small, ((1, 1), (1, 1)), mode='constant', constant_values=255)  # Pad to ensure we have a border around the checkerboard

        for i in range(len(marker_img_small)):
            for j in range(len(marker_img_small)):
                # Compute bottom-left corner of square
                offset_x = (i - grid_size / 2) * square_size
                offset_y = (j - grid_size / 2) * square_size

                origin = tvec + offset_x * x_axis + offset_y * y_axis

                # Create square in face plane for the marker overlay
                corners = np.array([
                    origin,
                    origin + square_size * x_axis,
                    origin + square_size * x_axis + square_size * y_axis,
                    origin + square_size * y_axis
                ])

                if marker_img_small[i, j] < 255:
                    color = [1.0, 1.0, 1.0, 1.0]
                else:
                    color = [0.0, 0.0, 0.0, 1.0]  # Black for the checkerboard background

                square = Poly3DCollection([corners], color=color, edgecolor='k')
                ax.add_collection3d(square)



    ax.set_title("Dodecahedron with Checkerboards on Each Face")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02])
    plt.tight_layout()
    plt.show()


# Load and visualize
#marker_extrinsics = load_aruco_group("aruco_group.json")
#plot_marker_frames(marker_extrinsics)

# Load and plot the combined data
faces_data = load_dodecahedron_with_markers("dodecahedron_with_markers.json")
#plot_dodecahedron_with_extrinsics(faces_data, axis_len=0.005)
#plot_dodecahedron_with_textured_markers(faces_data, axis_len=0.005)
plot_dodecahedron_with_checkerboards(faces_data, grid_size=5, square_size=0.002)
