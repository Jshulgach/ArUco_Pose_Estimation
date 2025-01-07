"""
Description:
    This is the `single_camera.py` script which performs pose estimation using a single camera. The user
    just needs to pass in the path to the calibration matrix and distortion coefficients (both saved
    as numpy files). Lastly, the ArUCo tag type must be specified.

Author:
    Jonathan Shulgach (jshulgac@andrew.cmu.edu)
Last Modified:
    01/05/2025

Usage:
    python dual_camera.py --yaml_dir config.yaml --stereo_calib stereo_calib.npz --type DICT_5X5_100 --visualize true
"""
import cv2
import numpy as np
import argparse
import utils

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", type=str, default='config.yaml', help="Path to yaml config file")
    ap.add_argument("--stereo_calib", type=str, default='stereo_calib.npz', help="Path to stereo calibration file")
    ap.add_argument("--aruco_type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    ap.add_argument("--visualize", type=bool, default=False, help="Enable/disable video output")
    args = vars(ap.parse_args())

    calib = utils.parse_config_file(args["yaml_dir"])
    stereo = np.load(args["stereo_calib"])

    # Open video streams
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    try:
        print("Running stereo pose estimation. Press 'q' to quit...")
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0 or not ret1:
                break

            # Perform the stereo pose estimation
            pose_0, pose_1 = utils.pose_estimation_dual_cameras(frame0, frame1,
                args['aruco_type'], calib, stereo)
            if args["visualize"]:
                cv2.imshow("Pose Estimation", np.hstack([pose_0, pose_1]))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
