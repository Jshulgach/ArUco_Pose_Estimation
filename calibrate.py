"""
Description:
    This is the `calibration.py` script which performs camera calibration using either a single
    camera or dual-camera setup. The user just needs to pass in the path to the configuration yaml
    file (or use the default settings). Enabling visualization (set to false by default) will show
    the checkerboard images as they are being processed.

Author:
    Jonathan Shulgach (jshulgac@andrew.cmu.edu)
Last Modified:
    01/05/2025

Usage:
    python calibration.py --yaml_dir calibration_settings.yaml --dual_camera True --visualize True
"""

import argparse
import utils
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", type=str, default='config.yaml', help="Path to yaml config file")
    ap.add_argument("--have_frames", type=bool, default=False, help="Allows skipping taking new photos")
    ap.add_argument("--dual_camera", type=bool, default=False, help="Set to True if using dual camera setup")
    ap.add_argument("--visualize", type=bool, default=False, help="To visualize each checkerboard image")
    args = ap.parse_args()
    calib = utils.parse_config_file(args.yaml_dir)

    if args.dual_camera:
        print("Performing dual camera calibration...")
        if not args.have_frames:
            print(" | Taking photos")
            utils.save_frames_dual_cameras(calib, args.visualize)

        # Perform stereo calibration
        print(" | Computing matrices")
        mtx0, dist0, mtx1, dist1, R, T, _, _ = utils.calibrate_dual_cameras(calib, args.visualize)

        # Save stereo calibration settings
        np.savez("stereo_calib.npz",
                 mtx0=mtx0, dist0=dist0,  # Camera 0 intrinsic matrix and distortion coefficients
                 mtx1=mtx1, dist1=dist1,  # Camera 1 intrinsic matrix and distortion coefficients
                 R=R, T=T)                 # Stereo calibration results
        print("Stereo calibration complete. Results saved to 'stereo_calib.npz'.")
    else:
        print("Performing single camera calibration...")
        if not args.have_frames:
            print(" | Taking photos")
            utils.save_frames_single_camera(calib, args.visualize)

        # Perform single camera calibration
        print(" | Computing matrices")
        _, mtx, dist = utils.calibrate_single_camera(calib, args.visualize)

        # Save calibration settings
        np.save("camera_intrinsics.npy", mtx)
        np.save("camera_distortion.npy", dist)
        print("Single-camera calibration complete, results saved")
