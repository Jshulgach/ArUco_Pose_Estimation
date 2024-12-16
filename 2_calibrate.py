'''
Sample Usage:-
python calibration.py --dir calibration_checkerboard/ --square_size 0.024
'''

import numpy as np
import cv2
import os
import argparse

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

    images = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.png', '.jpg')]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--img_dir", type=str, default='frames', help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-s", "--square_size", type=float, default=3.19, help="Length of one edge (in metres)")
    ap.add_argument("-w", "--width", type=int, default=7, help="Width of checkerboard (default=7)")
    ap.add_argument("-h", "--height", type=int, default=4, help="Height of checkerboard (default=4)")
    ap.add_argument("-v", "--visualize", type=str, default="True", help="To visualize each checkerboard image")
    args = vars(ap.parse_args())

    ret, mtx, dist, rvecs, tvecs = calibrate(args["img_dir"], args["square_size"], args["width"], args["height"], args["visualize"])

    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)
    print("Calibration complete, results saved")
