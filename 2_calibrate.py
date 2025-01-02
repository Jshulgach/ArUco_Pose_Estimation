'''
Sample Usage:-
python calibration.py --dir calibration_checkerboard/ --square_size 0.024
'''

import numpy as np
import cv2
import os
import argparse
from utils import parse_calibration_settings_file

def calibrate(dirpath, square_size, width, height, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if os.path.isdir(dirpath):
        print(f"{len(os.listdir(dirpath))} files found in '{dirpath}': ")
        for f in os.listdir(dirpath):
            print(f)
    else:
        print("Error: directory is not a valid path")
        return None

    images = [os.path.join(dirpath, f) for f in os.listdir(dirpath)]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

            if visualize:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (width, height), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    # Initialize arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", type=str, default='frames', help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("--visualize", type=str, action='store_true', help="To visualize each checkerboard image")
    args = vars(ap.parse_args())
    
    # Get settings
    calib = parse_calibration_settings_file(args["yaml_dir"])

    ret, mtx, dist, rvecs, tvecs = calibrate(
        calib["img_dir"], 
        calib['checkerboard_box_size_scale'],
        calib['checkerboard_column_vertices'],
        calib['checkerboard_row_vertices'], 
        args["visualize"]
        )

    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)
    print("Calibration complete, results saved")
