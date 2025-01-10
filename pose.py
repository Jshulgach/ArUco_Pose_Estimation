"""
Description:
    This is thef `single_camera.py` script which performs pose estimation using a single camera. The user
    just needs to pass in the path to the calibration matrix and distortion coefficients (both saved
    as numpy files). Lastly, the ArUCo tag type must be specified.

Author:
    Jonathan Shulgach (jshulgac@andrew.cmu.edu)
Last Modified:
    01/05/2025

Usage:
    python pose.py --yaml_dir config.yaml --type DICT_5X5_100
    # Pass other arguments as needed
"""
import cv2
import numpy as np
import argparse
import utils

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", type=str, default='config.yaml', help="Path to yaml config file")
    ap.add_argument("--dual_cameras", type=int, default=False, help="Number of cameras to use")
    ap.add_argument("--K_Matrix", type=str, default='camera_intrinsics.npy', help="Path to calibration matrix (numpy file)")
    ap.add_argument("--D_Coeff", type=str, default='camera_distortion.npy', help="Path to distortion coefficients (numpy file)")
    ap.add_argument("--stereo_calib", type=str, default='stereo_calib.npz', help="Path to stereo calibration file")
    ap.add_argument("--aruco_type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    ap.add_argument("--visualize", type=bool, default=False, help="Enable/disable video output")
    args = vars(ap.parse_args())
    calib = utils.parse_config_file(args["yaml_dir"])

    cap0 = cv2.VideoCapture(int(calib['camera0']))
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])

    if args["dual_cameras"]:
        cap1 = cv2.VideoCapture(int(calib['camera1']))
        stereo = np.load(args["stereo_calib"])
    try:
        print("Running {i} pose estimation. Press 'q' to quit...".format(i="stereo/dual camera" if args["dual_cameras"] else "single camera"))
        while True:

            ret0, frame0 = cap0.read()
            if not ret0:
                break
            if not args["dual_cameras"]:
                pose_image = utils.pose_estimation_single_camera(frame0, args['aruco_type'], calib, k, d)
            else:
                ret1, frame1 = cap1.read()
                if not ret1:
                    break

                # Perform the stereo pose estimation
                pose_0, pose_1 = utils.pose_estimation_dual_cameras(frame0, frame1, args['aruco_type'], calib, stereo)
                pose_image = np.hstack([pose_0, pose_1])

            if args["visualize"]:
                cv2.imshow('Pose Estimation', pose_image)
    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        cap0.release()
        if args["dual_cameras"]:
            cap1.release()
        cv2.destroyAllWindows()
