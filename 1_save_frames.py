import argparse
from utils import save_frames_single_camera, parse_calibration_settings_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", type=str, default='calibration_settings.yaml', help="Path to folder containing calibration settings")
    args = vars(ap.parse_args())
    
    calib = parse_calibration_settings_file(args["yaml_dir"])
    save_frames_single_camera(args["yaml_dir"], calib["img_dir"], calib["camera_id"])
