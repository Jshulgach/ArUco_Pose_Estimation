'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import argparse
from threading import Thread
from utils import ARUCO_DICT, draw_orientation

def pose_estimation(frame, arucoDict, arucoParams, matrix_coeff, distortion_coeff):
    """
    frame - Frame from the video stream
    arucoDict - ArUCo dictionary to use for detection
    arucoParams - Parameters for the ArUCo detection
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
    if ids is not None:
        for i in range(len(ids)):
            # Estimate pose for each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coeff, distortion_coeff)
            rot_mat = cv2.Rodrigues(rvec)[0]

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw the reference vectors and the arc between them
            frame = draw_orientation(frame, matrix_coeff, distortion_coeff, rot_mat, tvec, axis_length=0.05)

            # Display the euler angles on the frame
            angles = cv2.RQDecomp3x3(rot_mat)[0]
            cv2.putText(frame, f"Pitch: {angles[0]:.3f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Roll: {angles[1]:.3f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Yaw: {angles[2]:.3f} deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # write the xyz coordinates on the frame
            cv2.putText(frame, f"X: {tvec[0][0][0]:.3f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Y: {tvec[0][0][1]:.3f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Z: {tvec[0][0][2]:.3f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", type=str, default='calibration_matrix.npy', help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", type=str, default='distortion_coefficients.npy', help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--hide_video", action='store_false', help="Enable/disable video output")
    args = vars(ap.parse_args())

    aruco_dict_type = ARUCO_DICT[args["type"]]
    if not aruco_dict_type:
        raise ValueError("Invalid ArUCo tag type. Please refer to the documentation for valid types.")

    arucoDict = cv2.aruco.Dictionary_get(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters_create()
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])
    show_video = args["hide_video"]

    try:
        # Start the video stream
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            output_frame = pose_estimation(frame, arucoDict, arucoParams, k, d)
            if show_video:
                cv2.imshow('Pose Estimation', frame)
    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        cap.release()
        cv2.destroyAllWindows()
