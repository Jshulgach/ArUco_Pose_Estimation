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
    python single_camera.py --yaml_dir config.yaml --K_Matrix camera_intrinsics.npy --D_Coeff camera_distortion.npy --type DICT_5X5_100
"""
import cv2
import numpy as np
import argparse
import utils

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", type=str, default='config.yaml', help="Path to yaml config file")
    ap.add_argument("--K_Matrix", type=str, default='camera_intrinsics.npy', help="Path to calibration matrix (numpy file)")
    ap.add_argument("--D_Coeff", type=str, default='camera_distortion.npy', help="Path to distortion coefficients (numpy file)")
    ap.add_argument("--aruco_type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    ap.add_argument("--visualize", type=bool, default=False, help="Enable/disable video output")
    args = vars(ap.parse_args())

    calib = utils.parse_config_file(args["yaml_dir"])
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])

    cap = cv2.VideoCapture(0)
    print("Running pose estimation. Press 'q' to quit...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            pose_frame = utils.pose_estimation_single_camera(frame, args['aruco_type'], calib, k, d)
            if args["visualize"]:
                cv2.imshow('Pose Estimation', pose_frame)
    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        cap.release()
        cv2.destroyAllWindows()
