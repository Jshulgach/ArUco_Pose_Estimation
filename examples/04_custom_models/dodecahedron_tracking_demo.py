import cv2
import numpy as np
from gpt_aruco_pose_estimator import ArucoPoseEstimator
from gpt_pose_refiner import PoseRefiner
from gpt_predictive_filter import PredictivePoseFilter
from gpt_draw_overlay import draw_dodecahedron
from dodecahedron_model import CleanDodecahedronModel

def main():
    # === Load camera intrinsics ===
    K = np.load("camera_intrinsics.npy")
    D = np.load("camera_distortion.npy")

    # === Set parameters ===
    marker_size = 0.015  # in meters
    model_path = "dodecahedron_with_markers.json"
    #video_path = "WIN_20250404_11_11_34_Pro.mp4"

    # === Initialize estimator ===
    estimator = ArucoPoseEstimator(K, D, marker_size, model_path)
    predictor = PredictivePoseFilter(alpha=0.5)
    refiner = PoseRefiner(K, D)
    model = CleanDodecahedronModel(edge_length=0.025)

    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids = estimator.detect(frame)

        # Draw detected markers
        #if ids is not None:
        #    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvec, tvec, used_ids = estimator.estimate_pose(corners, ids)

        rvec_ref = None
        tvec_ref = None

        if rvec is not None:
            # Reconstruct obj_pts and img_pts from matched markers
            obj_pts = []
            img_pts = []
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in used_ids and marker_id in estimator.extrinsics:
                        r_local, t_local = estimator.extrinsics[marker_id]
                        size = marker_size / 2
                        obj = np.array([
                            [-size, size, 0],
                            [size, size, 0],
                            [size, -size, 0],
                            [-size, -size, 0]
                        ], dtype=np.float32)
                        R, _ = cv2.Rodrigues(r_local)
                        obj_world = (R @ obj.T).T + t_local
                        obj_pts.extend(obj_world)
                        img_pts.extend(corners[i][0])

            if len(obj_pts) >= 4:
                rvec_ref, tvec_ref, _ = refiner.refine_pose(
                    np.array(obj_pts), np.array(img_pts), rvec, tvec)
            else:
                rvec_ref, tvec_ref = rvec, tvec
        else:
            rvec_ref, tvec_ref = predictor.predict()

        predictor.update(rvec_ref, tvec_ref)

        # Draw coordinate frame if pose is valid
        if rvec is not None:
            draw_dodecahedron(frame, rvec_ref, tvec_ref, model, K, D)
            cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.03)
            cv2.putText(frame, f"Pose from IDs: {used_ids}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

        cv2.imshow("Dodecahedron Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()