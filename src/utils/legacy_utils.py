import os
import glob
import yaml
import cv2
import numpy as np
from collections import defaultdict, deque
from packaging.version import Version

# Dictionary mapping of ArUco names to OpenCV dictionary IDs
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def parse_config_file(filename):
    """ Load calibration settings from YAML file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Calibration settings file not found: {filename}")
    with open(filename, "r") as f:
        return yaml.safe_load(f)

def save_frames_single_camera(calib, visualize=False):
    """Capture frames for calibration using a single camera"""
    os.makedirs(calib["img_dir"], exist_ok=True)

    # get settings
    camera_device_id = calib['camera0']
    width = calib['frame_width']
    height = calib['frame_height']
    frames_to_save = calib['n_frames']
    view_resize = calib['view_resize']
    cooldown_time = calib['cooldown']

    # open video stream and change resolution.
    # Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv2.VideoCapture(camera_device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while saved_count < frames_to_save:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        frame_small = cv2.resize(frame, None, fx=1 / view_resize, fy=1 / view_resize)

        if start:
            cooldown -= 1
            if cooldown <= 0:  # save the frame when cooldown reaches 0.
                cv2.imwrite(os.path.join(calib["img_dir"], f"{calib['camera0']}_{saved_count}.png"), frame)
                saved_count += 1
                cooldown = cooldown_time

        if visualize:
            cv2.putText(
                frame_small,
                f"{'Saving' if start else 'Press SPACE to start'}: {saved_count}/{frames_to_save}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow("Calibration", frame_small)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                # if ESC is pressed at any time, the program will exit.
                break
            if k == 32:
                # Press spacebar to start data collection
                start = True

    cap.release()
    cv2.destroyAllWindows()

def calibrate_single_camera(calib, visualize=False):
    """ Perform single-camera calibration """
    pattern0 = f"{calib['camera0']}*.png"
    images = sorted(glob.glob(f"{calib['img_dir']}/{pattern0}"))
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows = calib['checkerboard_row_vertices']
    cols = calib['checkerboard_column_vertices']
    square_size = calib['checkerboard_box_size_scale']

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if visualize:
                # Draw and display corners
                img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist

def pose_estimation_single_camera(frame, aruco_type, aruco_size, matrix_coeff, distortion_coeff, show_text=False):
    """
    frame - Frame from the video stream
    arucoDict - ArUCo dictionary to use for detection
    arucoParams - Parameters for the ArUCo detection
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    markers - Marker rotation and translation vectors
    """
    aruco_dict_type = ARUCO_DICT[aruco_type]
    if not aruco_dict_type:
        raise ValueError("Invalid ArUCo tag type. Please refer to the documentation for valid types.")

    # Support for multiple cv2 versions
    if Version(cv2.__version__) >= Version("4.7.0"):
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    else:
        arucoDict = cv2.aruco.Dictionary_get(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters_create()
        detector = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    if detector:
        corners, ids, _ = detector.detectMarkers(image=gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    markers = {'pose':[], 'id': [], 'corners': []}  # Store marker poses and ids
    if ids is not None:
        for i in range(len(ids)):
            # Estimate pose for each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_size, matrix_coeff, distortion_coeff)
            rot_mat = cv2.Rodrigues(rvec)[0]
            markers['pose'].append([rvec,tvec])
            markers['id'].append(ids[i][0])  # Store the marker ID
            markers['corners'].append(corners[i])  # Store the corners for drawing

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw the id of the marker on the frame
            cv2.putText(frame, f"ID: {ids[i][0]}",
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10)),  # Position above the marker
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                        )

            if show_text:

                # Draw the reference vectors and the arc between them
                #frame = draw_orientation(frame, matrix_coeff, distortion_coeff, rot_mat, tvec, axis_length=0.05)
                cv2.drawFrameAxes(frame, matrix_coeff, distortion_coeff, rvec, tvec, 0.05, thickness=2)

                # Display the euler angles on the frame
                angles = cv2.RQDecomp3x3(rot_mat)[0]
                cv2.putText(frame, f"Pitch: {angles[0]:.3f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Roll: {angles[1]:.3f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Yaw: {angles[2]:.3f} deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # write the xyz coordinates on the frame
                cv2.putText(frame, f"X: {tvec[0][0][0]:.3f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Y: {tvec[0][0][1]:.3f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Z: {tvec[0][0][2]:.3f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame, markers

def save_frames_dual_cameras(calib, visualize=False):
    """Capture frames for calibration using a dual camera setup"""

    os.makedirs(calib["img_dir"], exist_ok=True)

    # Get calibration settings
    frame_height = calib['frame_height']
    frame_width = calib['frame_width']
    frames_to_save = calib['n_frames']
    cooldown_time = calib['cooldown']

    # Open cameras and set resolution
    cap0 = cv2.VideoCapture(calib['camera0'])
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap1 = cv2.VideoCapture(calib['camera1'])
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while saved_count < frames_to_save:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            raise RuntimeError("Failed to capture frame")

        frame0 = cv2.resize(frame0, (frame_width, frame_height))
        frame1 = cv2.resize(frame1, (frame_width, frame_height))

        if start:
            cooldown -= 1
            if cooldown <= 0:  # Save frames when cooldown reaches 0
                cv2.imwrite(os.path.join(calib["img_dir"], f"{calib['camera0']}_{saved_count}.png"), frame0)
                cv2.imwrite(os.path.join(calib["img_dir"], f"{calib['camera1']}_{saved_count}.png"), frame1)
                saved_count += 1
                cooldown = cooldown_time

        if visualize:
            cv2.putText(
                frame0,
                f"{'Saving' if start else 'Press SPACE to start'}: {saved_count}/{frames_to_save}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            frame0 = cv2.resize(frame0, None, fx=1 / calib['view_resize'], fy=1 / calib['view_resize'])
            frame1 = cv2.resize(frame1, None, fx=1 / calib['view_resize'], fy=1 / calib['view_resize'])
            cv2.imshow("Calibration", np.hstack([frame0, frame1]))
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # If ESC is pressed, exit the program
                break
            if k == 32:
                # Press spacebar to start data collection
                start = True

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def calibrate_dual_cameras(calib, visualize=False):
    """ Perform dual-camera stereo calibration """
    pattern0 = f"{calib['camera0']}*.png"
    c0_images = sorted(glob.glob(f"{calib['img_dir']}/{pattern0}"))
    pattern1 = f"{calib['camera1']}*.png"
    c1_images = sorted(glob.glob(f"{calib['img_dir']}/{pattern1}"))
    if not c0_images or not c1_images:
        raise FileNotFoundError(f"No images found in {calib['img_dir']} or {calib['img_dir']}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows = calib['checkerboard_row_vertices']
    cols = calib['checkerboard_column_vertices']
    square_size = calib['checkerboard_box_size_scale']

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size

    width = calib['frame_width']
    height = calib['frame_height']

    # Pixel coordinates of the checkerboard corners
    imgpoints0 = []
    imgpoints1 = []

    # Coordinates of the checkerboard corners in 3D space
    objpoints = []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray0 = cv2.cvtColor(cv2.imread(frame0), cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(cv2.imread(frame1), cv2.COLOR_BGR2GRAY)
        ret0, corners0 = cv2.findChessboardCorners(gray0, (rows, cols), None)
        ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, cols), None)

        if ret0 and ret1:
            corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

            if visualize:
                frame0 = cv2.drawChessboardCorners(cv2.imread(frame0), (cols, rows), corners0, ret0)
                frame1 = cv2.drawChessboardCorners(cv2.imread(frame1), (cols, rows), corners1, ret1)

                frame0 = cv2.resize(frame0, None, fx=1 / calib['view_resize'], fy=1 / calib['view_resize'])
                frame1 = cv2.resize(frame1, None, fx=1 / calib['view_resize'], fy=1 / calib['view_resize'])
                cv2.imshow("Calibration", np.hstack([frame0, frame1]))
                cv2.waitKey(0)

            objpoints.append(objp)
            imgpoints0.append(corners0)
            imgpoints1.append(corners1)

    cv2.destroyAllWindows()
    _, mtx_0, dist_0, _, _ = cv2.calibrateCamera(objpoints, imgpoints0, gray0.shape[::-1], None, None)
    _, mtx_1, dist_1, _, _ = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)

    ret, mtx0, dist0, mtx1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints0, imgpoints1, mtx_0, dist_0, mtx_1, dist_1, (gray0.shape[::-1]),
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )
    return mtx_0, dist_0, mtx_1, dist_1, R, T, E, F

def pose_estimation_dual_cameras(frame0, frame1, aruco_type, calib, stereo_data):
    """ Pose estimation function for dual-camera setup """
    aruco_dict_type = ARUCO_DICT[aruco_type]
    if not aruco_dict_type:
        raise ValueError("Invalid ArUCo tag type. Please refer to the documentation for valid types.")

    # Get stereo calibration data
    K0 = stereo_data['mtx0']
    D0 = stereo_data['dist0']
    K1 = stereo_data['mtx1']
    D1 = stereo_data['dist1']
    R = stereo_data['R']
    T = stereo_data['T']

    # Support for multiple cv2 versions
    if Version(cv2.__version__) >= Version("4.7.0"):
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    else:
        arucoDict = cv2.aruco.Dictionary_get(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters_create()
        detector = None

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Detect markers in both cameras
    if detector:
        corners0, ids0, _ = detector.detectMarkers(image=gray0)
        corners1, ids1, _ = detector.detectMarkers(image=gray1)
    else:
        corners0, ids0, _ = cv2.aruco.detectMarkers(gray0, arucoDict, parameters=arucoParams)
        corners1, ids1, _ = cv2.aruco.detectMarkers(gray1, arucoDict, parameters=arucoParams)

    if ids0 is not None and ids1 is not None:
        # Only process common marker IDs between both cameras
        common_ids = set(ids0.flatten()).intersection(ids1.flatten())

        for marker_id in common_ids:
            idx0 = list(ids0.flatten()).index(marker_id)
            idx1 = list(ids1.flatten()).index(marker_id)

            # Estimate the pose for each marker in each camera
            rvec0, tvec0, _ = cv2.aruco.estimatePoseSingleMarkers(corners0[idx0], calib['aruco_size'], K0, D0)
            rvec1, tvec1, _ = cv2.aruco.estimatePoseSingleMarkers(corners1[idx1], calib['aruco_size'], K1, D1)

            # Draw markers bounding boxes
            cv2.aruco.drawDetectedMarkers(frame0, corners0)
            cv2.aruco.drawDetectedMarkers(frame1, corners1)

            # Draw the frame Axes
            cv2.drawFrameAxes(frame0, K0, D0, rvec0, tvec0, calib['aruco_size'], thickness=2)
            cv2.drawFrameAxes(frame1, K1, D1, rvec1, tvec1, calib['aruco_size'], thickness=2)

            # Triangulate the 3D point from the two 2D points
            points4D = cv2.triangulatePoints(
                np.hstack((np.eye(3), np.zeros((3,1)))),  # Projection matrix for camera 0
                np.hstack((R, T)),  # Projection matrix for camera 1
                corners0[idx0].reshape(-1, 2).T,  # 2D point in camera 0
                corners1[idx1].reshape(-1, 2).T  # 2D point in camera 1
            )
            points3D = cv2.convertPointsFromHomogeneous(points4D.T).reshape(-1, 3)

            # Draw the 3d coordinates
            cv2.putText(frame0, f"X: {points3D[0][0]:.3f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame0, f"Y: {points3D[0][1]:.3f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame0, f"Z: {points3D[0][2]:.3f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame0, frame1

def compute_axes_vectors(matrix_coefficients, distortion_coefficients, rvec, tvec, axis_length=0.01):
    """
    Compute and draw the axes vectors corresponding to cv2.drawFrameAxes.
    """
    # Convert rvec to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Define 3D points for the axes in marker's local coordinate system
    origin = np.array([0, 0, 0], dtype=np.float32).reshape(-1, 3)
    x_axis = np.array([axis_length, 0, 0], dtype=np.float32).reshape(-1, 3)
    y_axis = np.array([0, axis_length, 0], dtype=np.float32).reshape(-1, 3)
    z_axis = np.array([0, 0, axis_length], dtype=np.float32).reshape(-1, 3)

    # Stack all points together
    axes_points = np.vstack([origin, x_axis, y_axis, z_axis])

    # Project 3D points to 2D image plane
    img_points, _ = cv2.projectPoints(axes_points, rvec, tvec, matrix_coefficients, distortion_coefficients)

    # Extract points
    origin_2d = tuple(img_points[0].ravel().astype(int))
    x_axis_2d = tuple(img_points[1].ravel().astype(int))
    y_axis_2d = tuple(img_points[2].ravel().astype(int))
    z_axis_2d = tuple(img_points[3].ravel().astype(int))

    # Return the 2D points for further processing if needed
    return origin_2d, x_axis_2d, y_axis_2d, z_axis_2d

def draw_orientation(frame, matrix_coeff, dist_coeff, rot_mat, tvec, axis_length=0.05):
    """
    Draws the reference and actual orientation vectors on the frame.
    Also draws an arc to represent the angle between them.
    """
    scale = 100  # Scale factor for the orientation vectors
    origin_2d, x_axis_2d, y_axis_2d, z_axis_2d = compute_axes_vectors(
        matrix_coeff, dist_coeff, rot_mat, tvec, axis_length=axis_length
    )

    # Draw marker axes
    cv2.line(frame, origin_2d, x_axis_2d, (0, 0, 255), 2)  # Red for X-axis
    cv2.line(frame, origin_2d, y_axis_2d, (0, 255, 0), 2)  # Green for Y-axis
    cv2.line(frame, origin_2d, z_axis_2d, (255, 0, 0), 2)  # Blue for Z-axis

    # Compute reference vector for yaw and draw axis
    y_ref_2d = np.array([origin_2d[0], origin_2d[1] - 1 * scale]).astype(int)  # Reference y-axis
    cv2.line(frame, origin_2d, y_ref_2d, (0, 255, 30), 2)  # Green for actual yaw

    # Get euler angles ( in degrees) and draw yaw arc
    angles = cv2.RQDecomp3x3(rot_mat)[0]
    cv2.ellipse(frame, origin_2d, (50, 50), 0, -90, angles[2] - 90, color=(0, 255, 255), thickness=2)
    return frame
