import argparse
from utils import save_frames_single_camera

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--img_dir", type=str, default='frames', help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-y", "--yaml_dir", type=str, default='calibration_settings.yaml', help="Path to folder containing calibration settings")
    ap.add_argument("-c", "--camera_id", type=str, default='camera0', help="Camera ID")
    args = vars(ap.parse_args())
    save_frames_single_camera(args["yaml_dir"], args["img_dir"], args["camera_id"])
