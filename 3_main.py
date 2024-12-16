'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import argparse
from threading import Thread
from utils import ARUCO_DICT, draw_orientation

class VideoStream:
    """Class to handle threaded video stream."""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.ret, self.frame = self.capture.read()
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

def pose_estimation(frame, detector, matrix_coeff, distortion_coeff):
    '''
    frame - Frame from the video stream
    detector - ArUCo detector object
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
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
    args = vars(ap.parse_args())

    aruco_dict_type = ARUCO_DICT[args["type"]]
    if not aruco_dict_type:
        raise ValueError("Invalid ArUCo tag type. Please refer to the documentation for valid types.")

    arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])

    # Start the video stream
    video_stream = VideoStream(0)
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        output_frame = pose_estimation(frame, detector, k, d)
        cv2.imshow('Pose Estimation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()