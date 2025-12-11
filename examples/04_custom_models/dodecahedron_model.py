# dodecahedron_model.py
import cv2
import numpy as np
from scipy.spatial import ConvexHull


class CleanDodecahedronModel:
    def __init__(self, edge_length=0.01):
        self.edge_length = edge_length
        self.vertices = self._generate_vertices()
        self.triangle_faces = self._compute_triangle_faces()
        self.pentagon_faces = self._define_pentagon_faces()
        self.aruco_id_order = ['0','1','3','4','5','6','7','8','9','10','11','12']


    def _define_pentagon_faces(self):
        # Canonical dodecahedron face indices (manually defined)
        return [
            [0, 8, 10, 2, 16],
            [12, 0, 16, 17, 1],
            [4, 8, 0, 12, 14],
            [6, 10, 8, 4, 18],
            [2, 10, 6, 15, 13],
            [16, 2, 13, 3, 17],
            [17, 3, 11, 9, 1],
            [14, 12, 1, 9, 5],
            [18, 4, 14, 5, 19],
            [15, 6, 18, 19, 7],
            [13, 15, 7, 11, 3],
            [5, 9, 11, 7, 19],
        ]

    def _generate_vertices(self):
        phi = (1 + np.sqrt(5)) / 2
        a, b = 1, 1 / phi
        points = [(x, y, z) for (x, y, z) in [
            ( a,  a,  a), ( a,  a, -a), ( a, -a,  a), ( a, -a, -a),
            (-a,  a,  a), (-a,  a, -a), (-a, -a,  a), (-a, -a, -a),
            ( 0,  b,  phi), ( 0,  b, -phi), ( 0, -b,  phi), ( 0, -b, -phi),
            ( b,  phi, 0), ( b, -phi, 0), (-b,  phi, 0), (-b, -phi, 0),
            ( phi, 0,  b), ( phi, 0, -b), (-phi, 0,  b), (-phi, 0, -b)
        ]]

        points = np.array(points)
        # Normalize scale to match desired edge length
        def compute_edge_length(vs):
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    d = np.linalg.norm(vs[i] - vs[j])
                    if d > 0.01:
                        return d

        scale = self.edge_length / compute_edge_length(points)
        return points * scale

    def _compute_triangle_faces(self):
        hull = ConvexHull(self.vertices)
        return hull.simplices

    def _face_normal(self, v0, v1, v2):
        n = np.cross(v1 - v0, v2 - v0)
        return n / np.linalg.norm(n)

    def get_face_centers(self):
        centers = []
        for face in self.pentagon_faces:
            face_verts = [self.vertices[i] for i in face]
            center = np.mean(face_verts, axis=0)
            centers.append(center)
        return np.array(centers)

    def get_marker_poses(self):
        poses = []
        for face in self.pentagon_faces:
            face_verts = self.vertices[face]
            center = np.mean(face_verts, axis=0)

            v0, v1, v2 = face_verts[:3]
            z_axis = np.cross(v1 - v0, v2 - v0)
            z_axis /= np.linalg.norm(z_axis)

            x_axis = v1 - v0
            x_axis -= np.dot(x_axis, z_axis) * z_axis
            x_axis /= np.linalg.norm(x_axis)

            y_axis = np.cross(z_axis, x_axis)

            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            t = center.reshape(3, 1)
            poses.append((R, t))
        return poses

    def get_marker_corners(self, marker_size):
        corners_3d = []
        half = marker_size / 2.0
        marker_local = np.array([
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0]
        ])

        poses = self.get_marker_poses()
        for R, t in poses:
            corners_world = (R @ marker_local.T).T + t.T
            corners_3d.append(corners_world)

        return corners_3d

    def get_marker_corners_with_ids(self, marker_size):
        all_corners = self.get_marker_corners(marker_size)
        return list(zip(range(len(all_corners)), all_corners))  # (id, corners)


class DodecahedronWithMarkers:
    def __init__(self, edge_length=0.017):
        self.edge_length = edge_length
        self.rotation_offsets = {
            0: -72,
            1: -110,
            3: -110,
            4: -110,
            5: -180,
            6: -180,
            7: -0,
            8: -0,
            9: -0,
            10: -0,
            11: -0,
            12: 0
        }
        self.aruco_id_order = ['0','1','3','4','5','6','7','8','9','10','11','12']
        self.vertices = self._generate_vertices()
        self.pentagon_faces = self._define_faces()
        self.extrinsics = self._compute_extrinsics()

    def _define_faces(self):
        return [
            [0, 8, 10, 2, 16],
            [12, 0, 16, 17, 1],
            [4, 8, 0, 12, 14],
            [6, 10, 8, 4, 18],
            [2, 10, 6, 15, 13],
            [16, 2, 13, 3, 17],
            [17, 3, 11, 9, 1],
            [14, 12, 1, 9, 5],
            [18, 4, 14, 5, 19],
            [15, 6, 18, 19, 7],
            [13, 15, 7, 11, 3],
            [5, 9, 11, 7, 19],
        ]

    def _generate_vertices(self):
        phi = (1 + np.sqrt(5)) / 2
        a, b = 1, 1 / phi
        base = np.array([
            ( a,  a,  a), ( a,  a, -a), ( a, -a,  a), ( a, -a, -a),
            (-a,  a,  a), (-a,  a, -a), (-a, -a,  a), (-a, -a, -a),
            ( 0,  b,  phi), ( 0,  b, -phi), ( 0, -b,  phi), ( 0, -b, -phi),
            ( b,  phi, 0), ( b, -phi, 0), (-b,  phi, 0), (-b, -phi, 0),
            ( phi, 0,  b), ( phi, 0, -b), (-phi, 0,  b), (-phi, 0, -b)
        ])
        def find_scale(vs):
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    d = np.linalg.norm(vs[i] - vs[j])
                    if d > 0.01:
                        return self.edge_length / d
        return base * find_scale(base)

    def _rotation_matrix_z(self, degrees):
        rad = np.deg2rad(degrees)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _compute_extrinsics(self):
        extrinsics_data = []
        for i, face in enumerate(self.pentagon_faces):
            aruco_id = int(self.aruco_id_order[i])
            face_verts = self.vertices[face]
            center = np.mean(face_verts, axis=0)

            x_axis = face_verts[1] - face_verts[0]
            x_axis /= np.linalg.norm(x_axis)
            v2 = face_verts[2] - face_verts[0]
            z_axis = np.cross(x_axis, v2)
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)

            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            R = R @ self._rotation_matrix_z(self.rotation_offsets.get(aruco_id, 0))
            rvec, _ = cv2.Rodrigues(R)

            extrinsics_data.append({
                "face_index": i,
                "aruco_id": aruco_id,
                "vertices": face_verts,
                "center": center,
                "extrinsics": np.concatenate([center, rvec.flatten()])
            })
        return extrinsics_data

