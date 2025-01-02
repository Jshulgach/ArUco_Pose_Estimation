import os
import yaml
import cv2
import numpy as np

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

def parse_calibration_settings_file(filename):
	""" Load calibration settings from YAML file"""
	if not os.path.exists(filename):
		raise FileNotFoundError(f"Calibration settings file not found: {filename}")

	with open(filename, 'r') as f:
		calibration_settings = yaml.safe_load(f)

	if 'camera0' not in calibration_settings:
		raise ValueError("Invalid calibration settings file. Missing camera0 key.")

	return calibration_settings

def save_frames_single_camera(yaml_filename, img_filepath, camera_name):
	"""Capture frames for calibration using a single camera"""
	os.makedirs(img_filepath, exist_ok=True)

	# get settings
	calibration_settings = parse_calibration_settings_file(yaml_filename)
	camera_device_id = calibration_settings[camera_name]
	width = calibration_settings['frame_width']
	height = calibration_settings['frame_height']
	frames_to_save = calibration_settings['mono_calibration_frames']
	view_resize = calibration_settings['view_resize']
	cooldown_time = calibration_settings['cooldown']

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
			if cooldown <= 0: # save the frame when cooldown reaches 0.
				cv2.imwrite(os.path.join(img_filepath, f"{camera_name}_{saved_count}.png"), frame)
				saved_count += 1
				cooldown = cooldown_time

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
