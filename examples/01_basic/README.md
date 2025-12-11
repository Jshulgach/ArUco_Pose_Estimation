# Basic Examples

This directory contains fundamental examples for getting started with ArUco marker detection and camera calibration.

## Files

### 1. `camera_calibration.py`
Calibrate your camera using a checkerboard pattern to obtain intrinsic parameters.

**Usage:**
```bash
python camera_calibration.py
```

**What it does:**
- Opens camera and captures checkerboard frames
- Computes camera matrix and distortion coefficients
- Saves calibration to `calibration_matrix.npy` and `distortion_coefficients.npy`

**Prerequisites:**
- Printed checkerboard pattern (default: 9x6 corners)
- Good lighting conditions
- Capture 20+ images from different angles

---

### 2. `generate_markers.py`
Generate ArUco markers for printing or display.

**Note:** This example demonstrates marker generation. The core utility is now in `src/utils/marker_generator.py`.

**Usage:**
```bash
python generate_markers.py --id 0 --size 200 --dict DICT_4X4_50
```

**Parameters:**
- `--id`: Marker ID to generate (default: 0)
- `--size`: Marker size in pixels (default: 200)
- `--dict`: ArUco dictionary type (default: DICT_4X4_50)
- `--output`: Output filename (default: marker_{id}.png)

**What it does:**
- Creates high-quality ArUco marker images
- Supports all OpenCV ArUco dictionaries
- Saves as PNG for easy printing

---

### 3. `basic_detection.py`
Detect ArUco markers in live camera feed or video files.

**Usage:**
```bash
# Live camera
python basic_detection.py

# Video file
python basic_detection.py --source path/to/video.mp4

# Image file
python basic_detection.py --source path/to/image.jpg
```

**What it does:**
- Detects ArUco markers in real-time
- Draws marker corners and IDs
- Shows FPS and detection statistics

**Prerequisites:**
- Camera or video file
- No calibration required (detection only)

---

### 4. `config_example.py`
Demonstrates the new configuration system for managing project settings.

**Usage:**
```bash
python config_example.py
```

**What it does:**
- Shows how to use `ProjectConfig` dataclass
- Demonstrates YAML configuration loading
- Validates configuration parameters
- Creates example config files

---

## Quick Start

1. **Calibrate your camera:**
   ```bash
   python camera_calibration.py
   ```

2. **Generate markers:**
   ```bash
   python generate_markers.py --id 0 --size 200
   python generate_markers.py --id 1 --size 200
   python generate_markers.py --id 2 --size 200
   ```

3. **Test detection:**
   ```bash
   python basic_detection.py
   ```

## Next Steps

After mastering these basics:
- Move to `02_single_marker/` for pose estimation
- Try `03_multi_marker/` for multi-marker tracking
- Explore `04_custom_models/` for dodecahedron tracking
- Check `05_advanced/` for optical flow and refinement

## Tips

- **Calibration:** Capture images with checkerboard at various angles and distances
- **Marker printing:** Print at high quality (300+ DPI) for best detection
- **Lighting:** Ensure even lighting without glare or shadows
- **Marker size:** Larger markers = easier detection from farther away
